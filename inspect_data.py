import pandas as pd
import json

# Load minute data
df1 = pd.read_csv('data/raw/EURUSD1.csv', sep=None, engine='python', header=None, names=['time','open','high','low','close','volume'], parse_dates=['time'])
# Load daily data
df1440 = pd.read_csv('data/raw/EURUSD1440.csv', sep=None, engine='python', header=None, names=['time','open','high','low','close','volume'], parse_dates=['time'])

print('=== Minute Data ===')
print('Rows:', len(df1))
print('Range:', df1['time'].min(), 'to', df1['time'].max())
print('Columns:', list(df1.columns))
print('Dtypes:', df1.dtypes.to_dict())
print('First 2 rows:')
print(df1.head(2).to_string())
print()
print('=== Daily Data ===')
print('Rows:', len(df1440))
print('Range:', df1440['time'].min(), 'to', df1440['time'].max())
print('First 2 rows:')
print(df1440.head(2).to_string())
print()

# Check for issues
print('=== Minute Data Quality ===')
print('Duplicates:', df1['time'].duplicated().sum())
print('Negative volume:', (df1['volume'] < 0).sum())
print('Missing values:', df1.isna().sum().sum())
# Check OHLC
ohlc_check = (df1['high'] < df1[['open','close','low']].max(axis=1)) | (df1['low'] > df1[['open','close','high']].min(axis=1))
print('Invalid OHLC:', ohlc_check.sum())

print()
print('=== Daily Data Quality ===')
print('Duplicates:', df1440['time'].duplicated().sum())
print('Negative volume:', (df1440['volume'] < 0).sum())
print('Missing values:', df1440.isna().sum().sum())
ohlc_check = (df1440['high'] < df1440[['open','close','low']].max(axis=1)) | (df1440['low'] > df1440[['open','close','high']].min(axis=1))
print('Invalid OHLC:', ohlc_check.sum())