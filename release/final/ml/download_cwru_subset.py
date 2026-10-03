#!/usr/bin/env python3
"""Fetch a small, public CWRU 12 kHz drive-end dataset subset."""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/s-whynot/CWRU-dataset/main/"
FILES = {
    "normal": ["Normal/97_Normal_0.mat", "Normal/98_Normal_1.mat", "Normal/99_Normal_2.mat", "Normal/100_Normal_3.mat"],
    "inner": ["12k_Drive_End_Bearing_Fault_Data/IR/007/105_0.mat", "12k_Drive_End_Bearing_Fault_Data/IR/007/106_1.mat", "12k_Drive_End_Bearing_Fault_Data/IR/007/107_2.mat", "12k_Drive_End_Bearing_Fault_Data/IR/007/108_3.mat"],
    "ball": ["12k_Drive_End_Bearing_Fault_Data/B/007/118_0.mat", "12k_Drive_End_Bearing_Fault_Data/B/007/119_1.mat", "12k_Drive_End_Bearing_Fault_Data/B/007/120_2.mat", "12k_Drive_End_Bearing_Fault_Data/B/007/121_3.mat"],
    "outer": ["12k_Drive_End_Bearing_Fault_Data/OR/007/@6/130@6_0.mat", "12k_Drive_End_Bearing_Fault_Data/OR/007/@6/131@6_1.mat", "12k_Drive_End_Bearing_Fault_Data/OR/007/@6/132@6_2.mat", "12k_Drive_End_Bearing_Fault_Data/OR/007/@6/133@6_3.mat"],
}

root = Path(__file__).parent / "data" / "cwru_12k_de"
for label, paths in FILES.items():
    out = root / label
    out.mkdir(parents=True, exist_ok=True)
    for rel in paths:
        dest = out / Path(rel).name
        if not dest.exists():
            print("Downloading", rel)
            urlretrieve(BASE + rel.replace("@", "%40").replace(" ", "%20"), dest)
print("Dataset ready:", root)
