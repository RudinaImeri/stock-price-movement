import streamlit as st
from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data

st.title("Data Overview")

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Nvidia (NVDA)": "NVDA"
}

label = st.selectbox("Choose a stock", STOCKS.keys())
symbol = STOCKS[label]

raw = load_market_data_from_api(symbol)
data = prepare_api_data(raw)

st.subheader(f"Preview — {label}")
st.dataframe(data.head(100), use_container_width=True)

st.write("Shape:", data.shape)
