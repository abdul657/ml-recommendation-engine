# ML Recommendation Engine

A Python recommendation system built around the MovieLens 100K ratings dataset. It compares user-based collaborative filtering, item-based collaborative filtering, and a matrix-factorization model, evaluates rating predictions with RMSE, and exposes trained models through a FastAPI service.

## Technology

- Python 3.10+
- pandas and NumPy for interaction data processing
- scikit-learn for cosine similarity and TruncatedSVD
- joblib for local model artifacts
- FastAPI, Pydantic and Uvicorn for inference endpoints
- pytest for model tests
- Docker for packaging the API

## Project workflow

~~~text
MovieLens ratings.csv
        |
        v
train.py
  |             |             |
User-based   Item-based    TruncatedSVD
 CF model      CF model      model
  |             |             |
  +-------------+-------------+
                |
          RMSE evaluation
                |
       saved_models/*.pkl
                |
                v
       FastAPI inference API
      /health /predict /recommend
~~~

## Dataset

The repository includes a MovieLens ratings sample at `data/raw/ratings.csv`. If it is missing, download and prepare MovieLens 100K:

~~~bash
python data/download_data.py
~~~

The downloader retrieves the public MovieLens 100K archive from GroupLens and converts its ratings file to the CSV schema expected by this project. Review the dataset terms and attribution at the GroupLens MovieLens website before redistribution.

## Set up

~~~bash
python -m venv .venv
~~~

Activate the environment:

~~~bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
~~~

Install the pinned-compatible dependencies:

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## Train and evaluate models

From the repository root:

~~~bash
python train.py
~~~

The script uses a reproducible 80/20 train/test split (random state 42), trains three recommenders, reports root mean squared error (RMSE), and writes artifacts to `saved_models/`.

Lower RMSE indicates lower prediction error on this particular held-out split. Results are dataset- and split-specific; do not interpret them as a production performance guarantee.

The generated model artifacts are ignored by Git because they are reproducible build outputs and may be large.

## Run the API

Train the models first, then start FastAPI:

~~~bash
uvicorn api.main:app --reload
~~~

- API overview: http://127.0.0.1:8000/
- Interactive Swagger docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

### Health

~~~http
GET /health
~~~

Returns the model artifacts successfully loaded by the API. A degraded status means no model artifact was available.

### Predict a rating

~~~http
POST /predict
Content-Type: application/json

{"user_id": 1, "item_id": 50, "model_type": "svd"}
~~~

Supported model types: `user_cf`, `item_cf`, and `svd`.

### Recommend items

~~~http
POST /recommend
Content-Type: application/json

{"user_id": 1, "n": 10, "model_type": "svd"}
~~~

For SVD recommendations, the API ranks items the user has not rated. The collaborative-filtering models use their own fitted user/item matrices.

## Docker

Train models on the host first so the generated artifacts exist in `saved_models/`. Then:

~~~bash
docker compose -f deployment/docker-compose.yml up --build
~~~

The compose file builds from the repository root, mounts the trained model artifacts and dataset into the container, and publishes port 8000.

## Tests

~~~bash
pytest -q
~~~

The matrix-factorization tests use a small in-memory interaction table and do not require the MovieLens dataset.

## Current scope and limitations

- This is a learning and portfolio project using an educational public ratings dataset.
- User- and item-based collaborative filtering build dense similarity matrices; they are not optimized for very large catalogs.
- The SVD implementation uses scikit-learn TruncatedSVD over a user-item residual matrix. It is not Surprise's SVD implementation.
- The API serves model inference only after model artifacts have been trained and saved.
- Authentication, user-level authorization, online feedback ingestion, model monitoring and cloud hosting are not implemented.

## License

No license is currently declared. Add a license before accepting external contributions or permitting reuse.
