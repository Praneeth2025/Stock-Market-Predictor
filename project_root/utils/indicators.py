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
