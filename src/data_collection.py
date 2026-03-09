import yfinance as yf
import pandas as pd

STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS"
]

MARKET_INDEX = "^NSEI"
SECTOR_INDEX = "^CNXIT"

START = "2014-01-01"
END = "2024-01-01"

data = []

print("Downloading stock data...")

for stock in STOCKS:

    df = yf.download(stock, start=START, end=END)

    df["stock"] = stock

    df = df.reset_index()

    data.append(df)

stocks_df = pd.concat(data)

market = yf.download(MARKET_INDEX, start=START, end=END)
market = market.reset_index()

sector = yf.download(SECTOR_INDEX, start=START, end=END)
sector = sector.reset_index()

market["index_close"] = market["Close"]
sector["sector_close"] = sector["Close"]

market = market[["Date","index_close"]]
sector = sector[["Date","sector_close"]]

stocks_df = stocks_df.merge(market,on="Date",how="left")
stocks_df = stocks_df.merge(sector,on="Date",how="left")

stocks_df.to_csv("raw_market_data.csv",index=False)

print("Raw dataset saved")