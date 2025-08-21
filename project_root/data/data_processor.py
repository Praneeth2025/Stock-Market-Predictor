import pandas as pd
import numpy as np
import sys
import os


sys.path.append(os.path.abspath("../utils"))
from indicators import calculate_ATR # type: ignore

data=pd.read_csv("xauusd_15m_data_180.csv")
data=calculate_ATR(data)

data["result"] = None
for i in range(len(data)):
    if pd.isna(data.loc[i, "ATR"]):
        data.loc[i, "result"] = "Hold"
        continue

    target_tp = data.loc[i, "open"] + 1.5 * data.loc[i, "ATR"]
    target_sl = data.loc[i, "open"] - 1.5 * data.loc[i, "ATR"]
    for j in range(i, min(i + 8, len(data))):
        # Check if TP hit
        if data.loc[j, "low"] <= target_tp <= data.loc[j, "high"]:
            data.loc[i, "result"] = "Buy"
            break
        # Check if SL hit
        if data.loc[j, "low"] <= target_sl <= data.loc[j, "high"]:
            data.loc[i, "result"] = "Sell"
            break

    if data.loc[i, "result"] is None:
        data.loc[i, "result"] = "Hold"

data.to_csv("processed_data.csv",index=False)