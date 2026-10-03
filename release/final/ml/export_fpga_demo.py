#!/usr/bin/env python3
"""Export two deterministic CWRU cases and fixed model tensors as FPGA ROMs."""
import json
from pathlib import Path
import numpy as np
from hardware_reference import LABELS, DATA, ART, load_window, run

ROOT=Path(__file__).parent
OUT=ROOT.parent/"quartus"/"de10_standard_hw"/"model_data"
OUT.mkdir(parents=True,exist_ok=True)

def write_hex(path,values,bits):
    mask=(1<<bits)-1
    path.write_text("".join(f"{int(x)&mask:0{bits//4}x}\n" for x in np.asarray(values).reshape(-1)))

def main():
    manifest=json.loads((ART/"accelerator_manifest.json").read_text())
    # SW[1]=0 is a normal recording; SW[1]=1 is an outer-race fault recording.
    choices=[("normal",0),("outer",3)]
    expected=[]
    inputs=[]
    for label,label_id in choices:
        x=load_window(sorted((DATA/label).glob("*.mat"))[-1],0)
        inputs.append(x)
        r=run(x,manifest)
        expected.append({"switch":label_id==3,"label":label,"class":r["prediction"],
                         "logits":r["logits"].astype(int).tolist()})
    write_hex(OUT/"demo_inputs.hex",np.concatenate(inputs),8)
    for layer in manifest["layers"]:
        write_hex(OUT/f"{layer['name']}_weights.hex",np.load(ART/layer["weight_file"]),8)
        write_hex(OUT/f"{layer['name']}_bias.hex",np.load(ART/layer["bias_file"]),32)
    (OUT/"expected_demo.json").write_text(json.dumps(expected,indent=2)+"\n")
    print(json.dumps(expected,indent=2))
if __name__=="__main__":
    main()
