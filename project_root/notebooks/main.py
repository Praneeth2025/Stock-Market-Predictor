import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

sys.path.append(os.path.abspath("../utils")) 
from indicators import calculate_RSI, calculate_MACD, calculate_SuperTrend #type: ignore



data = pd.read_csv("C:\\Users\\vamsi\\OneDrive\\Desktop\\Documents\\studies\\complete stock market predictor\\Stock-Market-Predictor\\project_root\\data\\processed_data.csv")

data.drop(["H-L", "H-PC", "L-PC", "TR","spread","real_volume"], axis=1, inplace=True)

#Feature Extraction
#Adding indicators
data=calculate_SuperTrend(data)
data=calculate_RSI(data)
data=calculate_MACD(data)

data['time'] = pd.to_datetime(data['time'])
#Adding time based indicators
data['Day of the week']=data['time'].dt.dayofweek
data['Is Weekend']=data['time'].dt.dayofweek>=4
data['Month']=data['time'].dt.day

