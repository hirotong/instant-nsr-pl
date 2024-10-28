import csv
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd

# root = "output/original/TanksAndTemple"
# root = "output/ablation/TanksAndTemple_residual"
# root = "output/ablation/GlossySynthetic_residual"
# root = "output/original/GlossySynthetic"
# root = "output/rotreal"
root = sys.argv[1]

DEFAULT_METRIC_DICT = {
    "PSNR": np.nan,
    "SSIM": np.nan,
    "LPIPS": np.nan,
    "completeness": np.nan,
    "accuracy": np.nan,
    "chamfer-L1": np.nan,
    "f-score": np.nan,
    "f-score-15": np.nan,
    "f-score-20": np.nan,
}

case_names = sorted(glob(os.path.join(root, "*")))
case_names = [c for c in case_names if os.path.isdir(c)]
summary = pd.DataFrame(columns=["Case"] + ["Method"] + list(DEFAULT_METRIC_DICT.keys()))
for case in case_names:
    print("Processing", case)
    case_name = os.path.basename(case).split("-")[-1]
    method_names = sorted(
        [m for m in os.listdir(case) if os.path.isdir(os.path.join(case, m))]
    )
    # summary = pd.DataFrame(columns=list(DEFAULT_METRIC_DICT.keys()) + ["Method"])
    for method in method_names:
        result_path = os.path.join(case, method, "save/eval.csv")
        method_name = method.split("@")[0]
        if not os.path.exists(result_path):
            summary = summary.append(
                {"Case": case_name, "Method": method_name}, ignore_index=True
            )
        else:
            result = pd.read_csv(result_path)
            result_dict = {
                k: result[k].values[0]
                if k in result.columns
                else DEFAULT_METRIC_DICT[k]
                for k in DEFAULT_METRIC_DICT.keys()
            }
            result_dict.update({"Case": case_name, "Method": method_name})
            summary = summary.append(result_dict, ignore_index=True)

    # summary[case] = summary

summary.to_csv(os.path.join(root, "summary.csv"), index=False)
# for case isummaryys.keys():
#     summary summaryys[case]
#     summary.to_csv(os.path.join(case, "summary.csv"), index=False)
