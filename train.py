from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from models.collaborative_filtering import ItemBasedCF, UserBasedCF
from models.matrix_factorization import MatrixFactorization

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "ratings.csv"
MODEL_DIR = ROOT / "saved_models"


def evaluate_model(model, test_data: pd.DataFrame) -> float:
    predictions = [
        model.predict(str(row.user_id), str(row.item_id))
        for row in test_data.itertuples(index=False)
    ]
    actuals = test_data["rating"].astype(float).tolist()
    return float(np.sqrt(mean_squared_error(actuals, predictions)))


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run python data/download_data.py first."
        )

    print("=== ML Recommendation Engine Training ===")
    ratings = pd.read_csv(DATA_PATH)
    required = {"user_id", "item_id", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing columns: {sorted(missing)}")

    ratings = ratings[["user_id", "item_id", "rating"]].dropna().copy()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="raise").astype(float)
    if len(ratings) < 5:
        raise ValueError("At least five ratings are required for a train/test split.")

    train_df, test_df = train_test_split(
        ratings, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Rows: {len(ratings)} | Train: {len(train_df)} | Test: {len(test_df)}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "user_based_cf.pkl": UserBasedCF(k=20),
        "item_based_cf.pkl": ItemBasedCF(k=20),
        "matrix_factorization.pkl": MatrixFactorization(n_factors=50, random_state=42),
    }

    metrics: dict[str, float] = {}
    for filename, model in models.items():
        print(f"Training {filename} ...")
        model.fit(train_df)
        rmse = evaluate_model(model, test_df)
        metrics[filename] = rmse
        joblib.dump(model, MODEL_DIR / filename)
        print(f"RMSE: {rmse:.4f} | saved to {MODEL_DIR / filename}")

    print("\n=== Evaluation summary (lower RMSE is better) ===")
    for name, rmse in metrics.items():
        print(f"{name}: {rmse:.4f}")


if __name__ == "__main__":
    main()
