import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.matrix_factorization import MatrixFactorization


def evaluate_model_cf(model, test_data):
    predictions = []
    actuals = []

    for _, row in test_data.iterrows():
        pred = model.predict(str(row["user_id"]), str(row["item_id"]))
        predictions.append(pred)
        actuals.append(row["rating"])

    return np.sqrt(mean_squared_error(actuals, predictions))


def evaluate_model_mf(model, test_data):
    predictions = []
    actuals = []

    for _, row in test_data.iterrows():
        pred = model.predict(str(row["user_id"]), str(row["item_id"]))
        predictions.append(pred)
        actuals.append(row["rating"])

    return np.sqrt(mean_squared_error(actuals, predictions))


def main():
    print("=== ML Recommendation Engine Training ===\n")

    # 1. Load MovieLens ratings.csv
    print("1. Loading MovieLens dataset...")
    df = pd.read_csv("data/raw/ratings.csv")

    # Ensure correct types
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)

    # 2. Train/test split
    print("2. Splitting dataset...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 3. Train User-Based CF
    print("\n3. Training User-Based CF...")
    user_cf = UserBasedCF(k=20)
    user_cf.fit(train_df)
    user_cf_rmse = evaluate_model_cf(user_cf, test_df)
    print(f"User-Based CF RMSE: {user_cf_rmse:.4f}")
    joblib.dump(user_cf, "saved_models/user_based_cf.pkl")

    # 4. Train Item-Based CF
    print("\n4. Training Item-Based CF...")
    item_cf = ItemBasedCF(k=20)
    item_cf.fit(train_df)
    item_cf_rmse = evaluate_model_cf(item_cf, test_df)
    print(f"Item-Based CF RMSE: {item_cf_rmse:.4f}")
    joblib.dump(item_cf, "saved_models/item_based_cf.pkl")

    # 5. Train Matrix Factorization (SVD)
    print("\n5. Training Matrix Factorization (SVD)...")
    mf = MatrixFactorization(n_factors=100, n_epochs=20)
    mf.fit(train_df)
    mf_rmse = evaluate_model_mf(mf, test_df)
    print(f"Matrix Factorization RMSE: {mf_rmse:.4f}")
    mf.save("saved_models/matrix_factorization.pkl")

    # 6. Summary
    print("\n=== Training Complete! ===")
    print("\nModel Performance Summary:")
    print(f"User-Based CF:        {user_cf_rmse:.4f}")
    print(f"Item-Based CF:        {item_cf_rmse:.4f}")
    print(f"Matrix Factorization: {mf_rmse:.4f}")


if __name__ == "__main__":
    main()
