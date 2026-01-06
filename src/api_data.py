import yfinance as yf


def load_market_data_from_api(symbol: str, start="2020-01-01"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = df.reset_index()

    df = df.rename(columns={
        "Date": "date",
        "Open": "price_open",
        "Close": "price_close",
        "High": "price_high",
        "Low": "price_low",
        "Volume": "volume"
    })

    df["exchange"] = symbol
    df["company_name"] = symbol

    return df
