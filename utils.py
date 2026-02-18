import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def load_interactions(filename: str = "interactions.csv") -> pd.DataFrame:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["user_id", "item_id", "rating"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=required_cols)

    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)

    return df


def train_test_split_interactions(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df, test_df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train_interactions.csv")
    test_path = os.path.join(PROCESSED_DIR, "test_interactions.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved test to {test_path}")


def run_data_pipeline(
    raw_filename: str = "interactions.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_interactions(raw_filename)
    df = clean_interactions(df)
    train_df, test_df = train_test_split_interactions(
        df, test_size=test_size, random_state=random_state
    )
    save_processed(train_df, test_df)
    return train_df, test_df
