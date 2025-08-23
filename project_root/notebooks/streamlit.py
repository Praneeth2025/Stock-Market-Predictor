import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import pickle
from datetime import datetime
import pickle
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from Execution_code import main_Execution_code

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
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Market Predictor",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <div style="background-color:white;padding:10px;">
        <h1 style="text-align:center;color:black;">📈 Stock Market Prediction App</h1>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------
# APP INFO
# -----------------------------
st.info(
    "This project combines technical analysis strategies with machine learning to predict stock movement outcomes — Profit, Loss, or Hold — based on a combination of Supertrend, RSI, ATR, and engineered features.\n"
    "**0 = Buy**, **1 = Hold**, **2 = Sell**\n\n"
    "Based on your pre-trained ensemble model stored in the repo."
)


# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict 🔮"):

    # --- 1. Fetch Data ---
    raw_df = get_yesterdays_data()
    predictions = main_Execution_code(raw_df)
    raw_df['Prediction'] = predictions
    # --- 2. Tabs for Open, High, Low, Close ---
    tab1, tab2, tab3, tab4 = st.tabs(["Open", "High", "Low", "Close"])

    with tab1:
        st.subheader("📊 Open Price Trend")
        fig = px.line(raw_df, x="datetime", y="open", title="XAUUSD Open Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📊 High Price Trend")
        fig = px.line(raw_df, x="datetime", y="high", title="XAUUSD High Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Low Price Trend")
        fig = px.line(raw_df, x="datetime", y="low", title="XAUUSD Low Price")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("📊 Close Price Trend")
        fig = px.line(raw_df, x="datetime", y="close", title="XAUUSD Close Price")
        st.plotly_chart(fig, use_container_width=True)

    # --- 4. Show Table ---
    st.subheader("📋 Raw Data & Predictions")
    st.dataframe(raw_df.tail(20))
