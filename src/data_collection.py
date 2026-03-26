import yfinance as yf
import pandas as pd
import time

STOCKS = [
    # Banking & Financials
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",

    # IT
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",

    # Energy & Oil
    "RELIANCE.NS",
    "ONGC.NS",
    "IOC.NS",
    "BPCL.NS",

    # Automobile
    "MARUTI.NS",
    "M&M.NS",
    "TATAMOTORS.NS",
    "HEROMOTOCO.NS",

    # FMCG / Consumer
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",

    # Infra / Cement
    "LT.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",

    # Pharma
    "SUNPHARMA.NS",
    "CIPLA.NS",

    # Telecom / Utilities
    "BHARTIARTL.NS",
    "NTPC.NS",
    "POWERGRID.NS"
]

MARKET_INDEX = "^NSEI"
SECTOR_INDEX = "^CNXIT"

START = "2019-01-01"
END = "2024-12-01"

data = pd.DataFrame()

print("Downloading stock data...")

for stock in STOCKS:

    df = yf.download(stock, start=START, end=END, progress=False)

    if df.empty:
        print("Skipped", stock)
        continue

    df = df.reset_index()

    name = stock.replace(".NS","")

    df = df[["Date","Open","High","Low","Close","Volume"]]

    df.rename(columns={
        "Open": f"{name}_Open",
        "High": f"{name}_High",
        "Low": f"{name}_Low",
        "Close": f"{name}_Close",
        "Volume": f"{name}_Volume"
    }, inplace=True)

    if data.empty:
        data = df
    else:
        data = pd.merge(data, df, on="Date", how="outer")

    print("Downloaded", stock)

    time.sleep(1)


print("Downloading market index...")
market = yf.download(MARKET_INDEX, start=START, end=END, progress=False)
market = market.reset_index()[["Date","Close"]]
market.rename(columns={"Close":"index_close"}, inplace=True)


print("Downloading sector index...")
sector = yf.download(SECTOR_INDEX, start=START, end=END, progress=False)
sector = sector.reset_index()[["Date","Close"]]
sector.rename(columns={"Close":"sector_close"}, inplace=True)


data = pd.merge(data, market, on="Date", how="left")
data = pd.merge(data, sector, on="Date", how="left")

data.to_csv("raw_market_data.csv", index=False)

print("Dataset saved")
print("Shape:", data.shape)