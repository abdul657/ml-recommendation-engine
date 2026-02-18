import os
import pandas as pd
import zipfile
import urllib.request

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_PATH = os.path.join(RAW_DIR, "ml-100k.zip")

def download_movielens():
    print("Downloading MovieLens 100K dataset...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(RAW_DIR)

    print("Converting u.data to ratings.csv...")
    data_path = os.path.join(RAW_DIR, "ml-100k", "u.data")
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    df.drop(columns=["timestamp"], inplace=True)
    df.to_csv(os.path.join(RAW_DIR, "ratings.csv"), index=False)

    print("Saved:", os.path.join(RAW_DIR, "ratings.csv"))
    print("MovieLens dataset ready!")

if __name__ == "__main__":
    download_movielens()
