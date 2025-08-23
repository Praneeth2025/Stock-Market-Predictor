import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
import pandas_ta.momentum.squeeze_pro as sp
sp.npNaN = np.nan
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import pandas_ta as ta
from ta.trend import MACD

def calculate_ATR(df, window=14):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    return df


def calculate_RSI(df, window=14):
    df['RSI'] = RSIIndicator(close=df['close'], window=window).rsi()
    return df

def calculate_SuperTrend(df, period=14, multiplier=3):
    print("HELLO")
    st = ta.supertrend(df['high'], df['low'], df['close'], length=period, multiplier=multiplier)
    st.columns = ['Supertrend', 'Supertrend_Direction', 'Supertrend_Trend', 'Supertrend_Source']
    
    df['Supertrend_Direction'] = st['Supertrend_Direction']
    return df




def calculate_MACD(df, window_fast=12, window_slow=26, window_sign=9, fillna=False):
    macd = MACD(
        close=df['close'],
        window_fast=window_fast,
        window_slow=window_slow,
        window_sign=window_sign,
        fillna=fillna
    )
    
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    return df




# -----------------------------
# 1. Load Model
# -----------------------------
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# 2. Fetch Data
# -----------------------------
def get_yesterdays_data(ticker_symbol="GC=F", interval="15m", period="5d"):
    ticker = yf.Ticker(ticker_symbol)

    # Yesterday in UTC
    yesterday = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")

    # Get data
    df = ticker.history(interval=interval, period=period)
    df = df[df.index.strftime("%Y-%m-%d") == yesterday].reset_index()

    # Lowercase column names
    df.columns = df.columns.str.lower()
    return df


# -----------------------------
# 3. Feature Engineering
# -----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = calculate_ATR(df)
    df = calculate_SuperTrend(df)
    df = calculate_RSI(df)
    df = calculate_MACD(df)
    df['time'] = pd.to_datetime(df['datetime'])
    df['Day of the week'] = df['time'].dt.dayofweek
    df['Month'] = df['time'].dt.day
    df['hour'] = df['time'].dt.hour
    df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
    high_liquidity_hours = [13, 14, 1, 12]
    df['high_liquidity'] = df['hour'].isin(high_liquidity_hours).astype(int)
    # Drop missing values
    df.dropna(inplace=True)
    print(df.shape)
    return df


# -----------------------------
# 4. Preprocessing (Scaling + PCA)
# -----------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Select features
    data_dropped=df.drop(["MACD_signal","MACD_diff","time"],axis=1)

    log_trns = FunctionTransformer(func=np.log1p)
    data_dropped['ATR'] = log_trns.fit_transform(data_dropped['ATR'])

    continuous_cols = ["ATR", "RSI","MACD","Day of the week","EMA_21","Month","hour"]  # example
    boolean_cols = ["Supertrend_Direction", "high_liquidity"]

    X_continuous = data_dropped[continuous_cols]
    X_boolean = data_dropped[boolean_cols]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_continuous)

    # PCA
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_scaled)

    # Final dataset
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2","PC3","PC4","PC5","PC6"])
    final_data = pd.concat([pca_df, X_boolean.reset_index(drop=True)], axis=1)
    print(final_data.shape)
    return final_data


# -----------------------------
# 5. Predict
# -----------------------------
def make_predictions( processed_data):
    with open("C:/Users/vamsi/OneDrive/Desktop/Documents/studies/complete stock market predictor/Stock-Market-Predictor/project_root/models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    predictions = model.predict(processed_data)
    return predictions


def main_Execution_code(raw_df):
    feature_df = add_features(raw_df)
    processed_df = preprocess_data(feature_df)
    predictions = make_predictions( processed_df)
    return predictions


