#!/usr/bin/env python3
"""Exact exported INT8 CNN reference and deterministic vector generator."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).parent
ART=ROOT/"artifacts"
DATA=ROOT/"data"/"cwru_12k_de"
LABELS=["normal","inner","ball","outer"]

def sat(value):
    return max(-128,min(127,int(value)))

def mac(x,w,b,shift,relu):
    """INT8 MAC plus INT32 bias, RTL arithmetic shift, saturation and ReLU."""
    out=np.empty((len(x),w.shape[1]),np.int8)
    accs=np.empty((len(x),w.shape[1]),np.int32)
    for row in range(len(x)):
        for col in range(w.shape[1]):
            acc=(int(b[col]) if col<len(b) else 0)+sum((int(x[row,k]) if k<x.shape[1] else 0)*int(w[k,col]) for k in range(w.shape[0]))
            acc=max(-(1<<31),min((1<<31)-1,acc))
            accs[row,col]=acc
            out[row,col]=sat((max(0,acc) if relu else acc)>>shift)
    return out,accs

def im2col(x,kernel):
    padded=np.pad(x,((kernel//2,kernel//2),(0,0)))
    # PyTorch Conv1d / exported weights use input-channel-major, then tap.
    return np.array([np.concatenate([padded[t:t+kernel,ch] for ch in range(x.shape[1])]) for t in range(len(x))],np.int8)

def load_window(path,index):
    data=loadmat(path)
    key=next(k for k in data if k.endswith("DE_time"))
    x=data[key].reshape(-1).astype(np.float32)[index*256:(index+1)*256]
    x=(x-x.mean())/(x.std()+1e-8)
    return np.clip(np.rint(x/.0625),-127,127).astype(np.int8)

def run(window,manifest):
    layers={layer["name"]:layer for layer in manifest["layers"]}
    def params(name):
        layer=layers[name]
        return layer,np.load(ART/layer["weight_file"]),np.load(ART/layer["bias_file"])
    info,w,b=params("conv1")
    conv1_input=im2col(window[:,None],5)
    conv1,conv1_acc=mac(conv1_input,w,b,info["requantize_right_shift"],True)
    pool=np.maximum(conv1[::2],conv1[1::2]).astype(np.int8)
    info,w,b=params("conv2")
    conv2_input=im2col(pool,3)
    conv2,conv2_acc=mac(conv2_input,w,b,info["requantize_right_shift"],True)
    # GAP averages already-quantized Conv2 values, so its deployment scale
    # remains the Conv2 output scale (0.25).  Use explicit signed truncation.
    gap=np.trunc(conv2.astype(np.int32).sum(0)/128).astype(np.int8)[None,:]
    info,w,b=params("classifier")
    logits8,classifier_acc=mac(gap,w,b,info["requantize_right_shift"],False)
    return {"input":window,"conv1_im2col":conv1_input,"conv1_acc":conv1_acc,
      "conv1":conv1,"pool":pool,"conv2_im2col":conv2_input,"conv2_acc":conv2_acc,
      "conv2":conv2,"gap":gap,"classifier_acc":classifier_acc,"logits8":logits8,
      "logits":logits8[0,:4],"prediction":int(np.argmax(logits8[0,:4]))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default=str(ROOT/"rtl_vectors"))
    parser.add_argument("--per-class",type=int,default=1)
    args=parser.parse_args()
    manifest=json.loads((ART/"accelerator_manifest.json").read_text())
    root=Path(args.output_dir); root.mkdir(parents=True,exist_ok=True)
    cases=[]
    for expected,label in enumerate(LABELS):
        recording=sorted((DATA/label).glob("*.mat"))[-1]
        for index in range(args.per_class):
            result=run(load_window(recording,index),manifest)
            name=f"{expected}_{label}_{index:03d}"
            folder=root/name; folder.mkdir(exist_ok=True)
            for key,value in result.items():
                if isinstance(value,np.ndarray):
                    np.save(folder/(key+".npy"),value)
            case={"name":name,"label":label,"expected_class":expected,
              "prediction":result["prediction"],"logits":result["logits"].astype(int).tolist()}
            (folder/"expected.json").write_text(json.dumps(case,indent=2)+"\n")
            cases.append(case)
    summary={"format":"phase3-rtl-reference-v1",
      "arithmetic":"INT8xINT8, INT32 accumulator/bias, arithmetic right shift, saturation, ReLU",
      "cases":cases,"agreement_with_labels":sum(c["expected_class"]==c["prediction"] for c in cases),
      "total_cases":len(cases)}
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
