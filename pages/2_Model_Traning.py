import streamlit as st
from sklearn.model_selection import train_test_split

from src.api_data import load_market_data_from_api
from src.data_preprocessing import prepare_api_data
from src.feature_engineering import encode_categorical
from src.model import train_model
from src.evaluate import evaluate_model

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Nvidia (NVDA)": "NVDA"
}

stock_label = st.selectbox(
    "Choose a stock",
    list(STOCKS.keys())
)

symbol = STOCKS[stock_label]


@st.cache_resource
def train_cached_model(X, y):
    return train_model(X, y)


if st.button("Train Model"):

    with st.spinner("Training..."):

        raw = load_market_data_from_api(symbol)
        df = prepare_api_data(raw)

        X = df.drop(columns=["result", "date"], errors="ignore")
        y = df["result"]

        X, encoders = encode_categorical(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        model = train_cached_model(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)

        st.session_state["model"] = model
        st.session_state["encoders"] = encoders
        st.session_state["feature_names"] = X.columns.tolist()
        st.session_state["last_stock_trained"] = symbol

    st.success("✅ Training complete")
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    st.metric("F1-score", f"{metrics['f1']:.3f}")
