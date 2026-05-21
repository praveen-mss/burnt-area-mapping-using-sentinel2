import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    train_size: float = 0.7,
    random_state: int = 42
):

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("Class distribution (train):")
    print(y_train.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test