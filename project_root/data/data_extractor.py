import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Login Details
LOGIN = 79826994
PASSWORD = "Ecstech9826994@"
SERVER = "Exness-MT5Trial8"
MT5_PATH = "C:\\Program Files\\MetaTrader 5 EXNESS\\terminal64.exe"

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15

# Initialize MT5
if not mt5.initialize(path=MT5_PATH, login=LOGIN, server=SERVER, password=PASSWORD):
    raise RuntimeError("MT5 initialize failed:", mt5.last_error())

# Calculate from date (1 month ago) and to date (now)
to_date = datetime.now()
from_date = to_date - timedelta(days=180)

# Get 15m historical data from MT5
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, from_date, to_date)

# Create DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

df.to_csv("xauusd_15m_data_180.csv", index=False)


# Shutdown MT5
mt5.shutdown()
