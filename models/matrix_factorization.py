from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class MatrixFactorization:
    """Rating predictor using TruncatedSVD over a user-item residual matrix."""

    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.model: TruncatedSVD | None = None
        self.user_item: pd.DataFrame | None = None
        self.predicted_ratings: pd.DataFrame | None = None
        self.user_means: dict[str, float] = {}
        self.global_mean = 3.0
        self.trained = False

    def fit(self, train_data: pd.DataFrame) -> "MatrixFactorization":
        required = {"user_id", "item_id", "rating"}
        missing = required.difference(train_data.columns)
        if missing:
            raise ValueError(f"Training data is missing required columns: {sorted(missing)}")
        if train_data.empty:
            raise ValueError("Training data cannot be empty.")

        data = train_data[["user_id", "item_id", "rating"]].copy()
        data["user_id"] = data["user_id"].astype(str)
        data["item_id"] = data["item_id"].astype(str)
        data["rating"] = pd.to_numeric(data["rating"], errors="raise").astype(float)
        if not np.isfinite(data["rating"]).all():
            raise ValueError("Ratings must be finite numbers.")

        self.global_mean = float(data["rating"].mean())
        self.user_item = data.pivot_table(
            index="user_id", columns="item_id", values="rating", aggfunc="mean"
        ).fillna(0.0)

        observed = self.user_item.where(self.user_item.ne(0.0))
        means = observed.mean(axis=1).fillna(self.global_mean)
        self.user_means = means.to_dict()

        residuals = self.user_item.sub(means, axis=0)
        residuals = residuals.where(self.user_item.ne(0.0), 0.0)
        matrix = residuals.to_numpy(dtype=np.float32)

        max_components = min(matrix.shape) - 1
        if max_components >= 1:
            components = min(self.n_factors, max_components)
            self.model = TruncatedSVD(
                n_components=components,
                n_iter=10,
                random_state=self.random_state,
            )
            latent_users = self.model.fit_transform(matrix)
            reconstructed = latent_users @ self.model.components_
        else:
            self.model = None
            reconstructed = np.zeros_like(matrix)

        prediction_matrix = reconstructed + means.to_numpy(dtype=np.float32)[:, None]
        prediction_matrix = np.clip(prediction_matrix, 1.0, 5.0)
        self.predicted_ratings = pd.DataFrame(
            prediction_matrix,
            index=self.user_item.index,
            columns=self.user_item.columns,
        )
        self.trained = True
        return self

    def predict(self, user_id: str | int, item_id: str | int) -> float:
        if not self.trained or self.predicted_ratings is None:
            raise ValueError("Model has not been trained or loaded.")
        user_key, item_key = str(user_id), str(item_id)
        if user_key in self.predicted_ratings.index and item_key in self.predicted_ratings.columns:
            return float(self.predicted_ratings.loc[user_key, item_key])
        return float(np.clip(self.user_means.get(user_key, self.global_mean), 1.0, 5.0))

    def recommend(
        self,
        user_id: str | int,
        all_items: list[str | int],
        rated_items: list[str | int],
        n: int = 10,
    ) -> list[tuple[str, float]]:
        if n < 1:
            raise ValueError("n must be at least 1.")
        rated = {str(item) for item in rated_items}
        candidates = [str(item) for item in all_items if str(item) not in rated]
        scored = [(item, self.predict(user_id, item)) for item in candidates]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:n]

    def save(self, filepath: str | Path) -> None:
        if not self.trained:
            raise ValueError("Train the model before saving it.")
        destination = Path(filepath)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    def load(self, filepath: str | Path) -> None:
        loaded = joblib.load(filepath)
        if not isinstance(loaded, MatrixFactorization) or not loaded.trained:
            raise ValueError("File does not contain a trained MatrixFactorization model.")
        self.__dict__.update(loaded.__dict__)
