import pandas as pd
import numpy as np


def prepare_api_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["exchange", "date"])

    # Price change
    df["return"] = (
        df.groupby("exchange")["price_close"]
        .pct_change()
    )

    # Target:
    # 0 = Sell, 1 = Hold, 2 = Buy
    df["result"] = np.select(
        [
            df["return"] < -0.005,
            df["return"].between(-0.005, 0.005),
            df["return"] > 0.005
        ],
        [0, 1, 2]
    )

    # Simple features
    df["price_range"] = df["price_high"] - df["price_low"]
    df["price_change"] = df["price_close"] - df["price_open"]

    return df.dropna()
