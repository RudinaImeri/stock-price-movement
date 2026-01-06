from sklearn.preprocessing import LabelEncoder


def encode_categorical(df, encoders=None):
    df = df.copy()

    cat_cols = df.select_dtypes(include=["object"]).columns

    if encoders is None:
        encoders = {}

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
            else:
                # unseen column safety
                df[col] = 0

    return df, encoders
