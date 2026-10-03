#!/usr/bin/env python3
"""Train CWRU CNN; export shift-only INT8 matrices for the existing accelerator."""
import json, math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).parent; DATA=ROOT/"data"/"cwru_12k_de"; OUT=ROOT/"artifacts"; OUT.mkdir(exist_ok=True)
LABELS=["normal","inner","ball","outer"]; WINDOW=256; N=128; LANE=8
class TinyCNN(nn.Module):
 def __init__(self):
  super().__init__(); self.c1=nn.Conv1d(1,8,5,padding=2); self.c2=nn.Conv1d(8,8,3,padding=1); self.fc=nn.Linear(8,4)
 def forward(self,x): return self.fc(F.adaptive_avg_pool1d(F.relu(self.c2(F.max_pool1d(F.relu(self.c1(x)),2))),1).squeeze(-1))
def read(p):
 d=loadmat(p); return d[next(k for k in d if k.endswith("DE_time"))].reshape(-1).astype(np.float32)
def data(split):
 xs=[]; ys=[]
 for y,label in enumerate(LABELS):
  fs=sorted((DATA/label).glob("*.mat"))
  if len(fs)!=4: raise RuntimeError(f"Need 4 {label} files; found {len(fs)}")
  for p in (fs[:-1] if split=="train" else fs[-1:]):
   s=read(p)
   for i in range(N):
    x=s[i*WINDOW:(i+1)*WINDOW]; xs.append((x-x.mean())/(x.std()+1e-8)); ys.append(y)
 return torch.tensor(np.asarray(xs))[:,None,:],torch.tensor(ys)
def scl(x):
 return float(2**math.ceil(math.log2(max(float(x.detach().abs().max())/127,2**-24))))
def qi(x,s): return torch.clamp(torch.round(x/s),-127,127).to(torch.int32)
def rq(a,asc,osc,relu):
 ratio=osc/asc; sh=int(round(math.log2(ratio)))
 if not -31<=sh<=31 or not math.isclose(ratio,2**sh,rel_tol=1e-6): raise RuntimeError("not shift representable")
 if sh>0:
  r=1<<(sh-1); z=torch.where(a>=0,(a+r)>>sh,-(((-a)+r)>>sh))
 elif sh<0: z=a<<(-sh)
 else: z=a
 if relu: z=torch.maximum(z,torch.zeros_like(z))
 return torch.clamp(z,-127,127).to(torch.int32),sh
def calib(m,x):
 with torch.no_grad():
  a=F.relu(m.c1(x)); b=F.relu(m.c2(F.max_pool1d(a,2))); g=F.adaptive_avg_pool1d(b,1).squeeze(-1); l=m.fc(g)
 return {"input":scl(x),"conv1_output":scl(a),"conv2_output":scl(b),"gap_output":scl(b),"logits":scl(l)}
def integer(m,x,s):
 w1,w2,w3=scl(m.c1.weight),scl(m.c2.weight),scl(m.fc.weight)
 x1,z1=qi(x,s["input"]),qi(m.c1.weight,w1); b1=torch.round(m.c1.bias/(s["input"]*w1)).to(torch.int32)
 a=F.conv1d(x1.float(),z1.float(),b1.float(),padding=2).round().to(torch.int32); a,h1=rq(a,s["input"]*w1,s["conv1_output"],True)
 z2=qi(m.c2.weight,w2); b2=torch.round(m.c2.bias/(s["conv1_output"]*w2)).to(torch.int32)
 b=F.conv1d(F.max_pool1d(a.float(),2),z2.float(),b2.float(),padding=1).round().to(torch.int32); b,h2=rq(b,s["conv1_output"]*w2,s["conv2_output"],True)
 g=torch.round(b.sum(2).float()/b.shape[2]).to(torch.int32); z3=qi(m.fc.weight,w3); b3=torch.round(m.fc.bias/(s["conv2_output"]*w3)).to(torch.int32)
 l=F.linear(g.float(),z3.float(),b3.float()).round().to(torch.int32); l,h3=rq(l,s["conv2_output"]*w3,s["logits"],False)
 return l,[(z1,b1,w1,h1),(z2,b2,w2,h2),(z3,b3,w3,h3)]
def emit(name,w,b,ws,sh,ic,k,oc,ins,outs,relu,length):
 if k is None: mat,K=w.numpy().T,ic
 else: mat,K=w.numpy().transpose(1,2,0).reshape(ic*k,oc),ic*k
 PK,PO=math.ceil(K/LANE)*LANE,math.ceil(oc/LANE)*LANE; pad=np.zeros((PK,PO),np.int8); pad[:K,:oc]=mat.astype(np.int8)
 np.save(OUT/f"{name}_weights_kxoc_int8.npy",pad); np.save(OUT/f"{name}_bias_accumulator_int32.npy",b.numpy().astype(np.int32))
 return {"name":name,"operation":"conv1d_im2col" if k else "dense","kernel":k,"logical_k":K,"physical_k":PK,"logical_output_channels":oc,"physical_output_channels":PO,"input_channel_tiles":PK//LANE,"output_channel_tiles":PO//LANE,"output_length":length,"weight_file":f"{name}_weights_kxoc_int8.npy","weight_layout":"[K, output_channel]","bias_file":f"{name}_bias_accumulator_int32.npy","accumulator_bits":32,"input_scale":ins,"weight_scale":ws,"accumulator_scale":ins*ws,"output_scale":outs,"zero_point":0,"requantize_shift":sh,"requantize_right_shift":max(sh,0),"requantize_left_shift":max(-sh,0),"relu":relu}
def main():
 torch.manual_seed(7); np.random.seed(7); xt,yt=data("train"); xv,yv=data("test"); m=TinyCNN(); opt=torch.optim.Adam(m.parameters(),lr=1e-3)
 for _ in range(12):
  for x,y in DataLoader(TensorDataset(xt,yt),batch_size=64,shuffle=True): opt.zero_grad(); F.cross_entropy(m(x),y).backward(); opt.step()
 m.eval()
 with torch.no_grad(): fp=float((m(xv).argmax(1)==yv).float().mean())
 s=calib(m,xt); logs,ls=integer(m,xv,s); i8=float((logs.argmax(1)==yv).float().mean()); torch.save(m.state_dict(),OUT/"fp32_model.pt")
 a,b,wa,h=ls[0]; c,d,wc,j=ls[1]; e,f,we,q=ls[2]
 result={"format":"phase3-de10-int8-v1","dataset":"CWRU 12 kHz drive-end bearing vibration; recording-level holdout","labels":LABELS,"window_samples":WINDOW,"normalization":"per-window zero mean/unit deviation","fp32_accuracy":fp,"integer_int8_accuracy":i8,"activation_spec":{n:{"scale":v,"zero_point":0,"dtype":"int8","scale_type":"power_of_two"} for n,v in s.items()},"execution":["Conv1: 256 im2col vectors; one 8x8 tile each.","MaxPool1D(2): HPS or board wrapper.","Conv2: 128 im2col vectors; three 8x8 K-tiles; combine INT32 sums before bias/shift/ReLU.","GlobalAveragePool1D(128): HPS or board wrapper.","Classifier: one 8x8 tile; only lanes 0..3 are logits."],"layers":[emit("conv1",a,b,wa,h,1,5,8,s["input"],s["conv1_output"],True,256),emit("conv2",c,d,wc,j,8,3,8,s["conv1_output"],s["conv2_output"],True,128),emit("classifier",e,f,we,q,8,None,4,s["conv2_output"],s["logits"],False,1)]}
 for p in ("accelerator_manifest.json","metrics_and_quantization.json"): (OUT/p).write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"fp32_accuracy":fp,"integer_int8_accuracy":i8,"layers":result["layers"]},indent=2))
if __name__=="__main__": main()
