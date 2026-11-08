import pandas as pd
import pytest

from models.matrix_factorization import MatrixFactorization


@pytest.fixture
def ratings():
    return pd.DataFrame([
        {"user_id": "1", "item_id": "10", "rating": 5.0},
        {"user_id": "1", "item_id": "11", "rating": 4.0},
        {"user_id": "1", "item_id": "12", "rating": 1.0},
        {"user_id": "2", "item_id": "10", "rating": 4.0},
        {"user_id": "2", "item_id": "11", "rating": 5.0},
        {"user_id": "2", "item_id": "12", "rating": 2.0},
        {"user_id": "3", "item_id": "10", "rating": 1.0},
        {"user_id": "3", "item_id": "11", "rating": 2.0},
        {"user_id": "3", "item_id": "12", "rating": 5.0},
    ])


def test_fit_and_predict(ratings):
    model = MatrixFactorization(n_factors=2).fit(ratings)
    score = model.predict("1", "10")
    assert model.trained is True
    assert 1.0 <= score <= 5.0


def test_recommend_excludes_rated_items(ratings):
    model = MatrixFactorization(n_factors=2).fit(ratings)
    recommendations = model.recommend("1", ["10", "11", "12", "13"], ["10", "11"], n=5)
    assert all(item not in {"10", "11"} for item, _ in recommendations)
    assert {item for item, _ in recommendations} == {"12", "13"}


def test_save_and_load_round_trip(ratings, tmp_path):
    model = MatrixFactorization(n_factors=2).fit(ratings)
    path = tmp_path / "models" / "matrix_factorization.pkl"
    model.save(path)

    restored = MatrixFactorization()
    restored.load(path)
    assert restored.predict("1", "10") == pytest.approx(model.predict("1", "10"))


def test_fit_rejects_empty_data():
    with pytest.raises(ValueError, match="cannot be empty"):
        MatrixFactorization().fit(pd.DataFrame(columns=["user_id", "item_id", "rating"]))
