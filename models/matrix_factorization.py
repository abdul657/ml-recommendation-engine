import joblib

class MatrixFactorization:
    """
    Lightweight Matrix Factorization wrapper for inference only.
    This version does NOT train inside Docker. It only loads a pretrained Surprise SVD model.
    """

    def __init__(self):
        self.model = None
        self.trained = False

    def load(self, filepath):
        """Load pretrained SVD model"""
        self.model = joblib.load(filepath)
        self.trained = True
        print(f"Matrix Factorization model loaded from {filepath}")

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        if not self.trained:
            raise ValueError("MatrixFactorization model not loaded")

        prediction = self.model.predict(user_id, item_id)
        return prediction.est

    def recommend(self, user_id, all_items, rated_items, n=10):
        """Recommend top n items for a user"""
        if not self.trained:
            raise ValueError("MatrixFactorization model not loaded")

        # Items user has not rated
        unrated_items = [item for item in all_items if item not in rated_items]

        predictions = []
        for item_id in unrated_items:
            score = self.predict(user_id, item_id)
            predictions.append((item_id, score))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:n]
