import streamlit as st
import pandas as pd
import numpy as np

from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data
from src.feature_engineering import encode_categorical

st.set_page_config(page_title="Market Predictions", layout="wide")
st.title("Market Predictions")

STOCKS = {
    "AAPL": {"name": "Apple"},
    "MSFT": {"name": "Microsoft"},
    "GOOGL": {"name": "Google"},
    "AMZN": {"name": "Amazon"},
    "TSLA": {"name": "Tesla"},
    "META": {"name": "Meta"},
    "NVDA": {"name": "Nvidia"},
}

required = ["model", "encoders", "feature_names", "last_stock_trained"]
missing = [k for k in required if k not in st.session_state]

if missing:
    st.warning("Please train the model first.")
    st.stop()

model = st.session_state["model"]
encoders = st.session_state["encoders"]
feature_names = st.session_state["feature_names"]
trained_symbol = st.session_state["last_stock_trained"]

st.info(f"Model trained on: **{trained_symbol}**")

raw = load_market_data_from_api(trained_symbol)
df = prepare_api_data(raw)

X = df.drop(columns=["result", "date"], errors="ignore")
X, _ = encode_categorical(X, encoders)
X = X.reindex(columns=feature_names, fill_value=0)

preds = model.predict(X)

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)
    confidence = np.max(probs, axis=1)
else:
    confidence = np.ones(len(preds))

class_map = {0: "Sell", 1: "Hold", 2: "Buy"}

price_close = df["price_close"]
price_open = df["price_open"]

company_names = [v["name"] for v in STOCKS.values()]

table = pd.DataFrame({
    "Symbol": df["exchange"],
    "Name": df["exchange"].map(lambda s: STOCKS[s]["name"]),
    "Price": price_close.round(2),
    "Change": (price_close - price_open).round(2),
    "Change %": ((price_close - price_open) / price_open * 100).round(2),
    "Volume": df["volume"],
    "Prediction": [class_map[int(p)] for p in preds],
    "Confidence": confidence.round(3)
})

view = st.radio(
    "View",
    ["Most Active", "Top Gainers", "Top Losers"],
    horizontal=True
)

if view == "Most Active":
    table = table.sort_values("Volume", ascending=False)
elif view == "Top Gainers":
    table = table.sort_values("Change %", ascending=False)
else:
    table = table.sort_values("Change %", ascending=True)


def color_change(val):
    if isinstance(val, (int, float)):
        return "color: green" if val > 0 else "color: red" if val < 0 else ""
    return ""


rows = st.slider("Rows to display", 20, 200, 50)

st.dataframe(
    table.head(rows)
         .style
         .applymap(color_change, subset=["Change", "Change %"])
         .format({
             "Price": "{:.2f}",
             "Change %": "{:.2f}%",
             "Volume": "{:,}",
             "Confidence": "{:.3f}"
         }),
    use_container_width=True,
    height=550
)

st.download_button(
    "⬇ Download predictions.csv",
    data=table.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv",
    key="download_yahoo_style"
)
