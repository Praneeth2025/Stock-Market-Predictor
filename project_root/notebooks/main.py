import pandas as pd
import numpy as np
import sys
import os
import seaborn as sns

from sklearn.preprocessing import FunctionTransformer, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath("../utils")) 
from indicators import calculate_RSI, calculate_MACD, calculate_SuperTrend  # type: ignore

a=0


# Load Data
data = pd.read_csv("C:\\Users\\vamsi\\OneDrive\\Desktop\\Documents\\studies\\complete stock market predictor\\Stock-Market-Predictor\\project_root\\data\\processed_data.csv")

# Drop unwanted columns early
data.drop(["H-L", "H-PC", "L-PC", "TR", "spread"], axis=1, inplace=True)



# Apply Indicator Functions
data = calculate_SuperTrend(data)
data = calculate_RSI(data)
data = calculate_MACD(data)

# Parse time and create time-based features
data['time'] = pd.to_datetime(data['time'])
data['Day of the week'] = data['time'].dt.dayofweek
data['Is Weekend'] = data['time'].dt.dayofweek >= 4
data['Month'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour

# Create custom liquidity feature
high_liquidity_hours = [13, 14, 1, 12]
data['high_liquidity'] = data['hour'].isin(high_liquidity_hours).astype(int)

# Drop missing values
data.dropna(inplace=True)



# Drop entries with low volume day
data = data[data['Day of the week'] < 5]




# Remove unnecessary columns
data_reduced = data.iloc[:, 7:]




# Apply log transformation for skewed features
log_trns = FunctionTransformer(func=np.log1p)
data['ATR'] = log_trns.fit_transform(data['ATR'])



# Outlier Removal Function
def hybrid_outler_removal(df, cols, extreme_thresh=2):
    clean_df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        lower_extreme = Q1 - extreme_thresh * IQR
        upper_extreme = Q3 + extreme_thresh * IQR

        # Remove extreme outliers
        clean_df = clean_df[(clean_df[col] >= lower_extreme) & (clean_df[col] <= upper_extreme)]
        count = df.shape[0] - clean_df.shape[0]
        print(count)

        # Clip mild outliers
        clean_df[col] = clean_df[col].clip(lower=lower_bound, upper=upper_bound)
        
    return clean_df

# Apply outlier removal
data_reduced = hybrid_outler_removal(data_reduced, ["MACD"])

x = data_reduced.drop(['result', 'MACD_signal', 'MACD_diff'], axis=1)
y = data_reduced['result']

"""
Dimensionality:
As per the information from the Elbow method, elbow is forming at 4 but the cumulative variance reached 0.95 at 6..
so, i am taking the dimensions as 5 for a balance
"""

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Standardize features
standardizer = StandardScaler()
x_standardized = standardizer.fit_transform(x)

# Apply PCA
pca = PCA(n_components=6)
x_reduced_dim = pca.fit_transform(x_standardized)




#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.2)
