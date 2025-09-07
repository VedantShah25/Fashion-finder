import pandas as pd
import os


DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl"
DATA_PATH = "data/swift-style-embeddings.pkl"

def load_dataset():
    """Download and load the dataset into a Pandas DataFrame."""
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(DATA_PATH):
        os.system(f"wget -O {DATA_PATH} {DATA_URL}")
    df = pd.read_pickle(DATA_PATH)
    return df