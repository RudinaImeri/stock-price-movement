import streamlit as st
import pandas as pd

st.title("Feature Importance")

if "model" not in st.session_state or "feature_names" not in st.session_state:
    st.warning("Please train the model first.")
    st.stop()

model = st.session_state["model"]
feature_names = st.session_state["feature_names"]

if not hasattr(model, "feature_importances_"):
    st.error("This model does not support feature importance.")
    st.stop()

trained_symbol = st.session_state["last_stock_trained"]

st.info(f"Model trained on: **{trained_symbol}**")


fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_,
    "OriginalIndex": range(len(feature_names))
})

fi_top = fi.sort_values("Importance", ascending=False).head(40)

fi_top_original_order = fi_top.sort_values("OriginalIndex")

st.subheader("Top Features")
st.dataframe(
    fi_top_original_order[["Feature", "Importance"]],
    use_container_width=True
)

st.subheader("Importance Chart")
st.bar_chart(
    fi_top_original_order.set_index("Feature")["Importance"]
)
