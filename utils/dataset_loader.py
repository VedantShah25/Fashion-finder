# utils/dataset_loader.py
import pandas as pd
import os
import requests

from utils.regenerate_embeddings import regenerate_embeddings

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/95eJ0YJVtqTZhEd7RaUlew/processed-swift-style-with-embeddings.pkl"
ORIGINAL_PATH = "data/swift-style-embeddings.pkl"
DATA_PATH_512 = "data/swift-style-embeddings-512.pkl"

def download_file(url, local_path):
    """Download file from a URL using requests (chunked for safety)."""
    print(f"[INFO] Downloading {url} → {local_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("[INFO] Download complete.")

def load_dataset():
    """Load dataset with 512-dim embeddings, regenerating if needed."""
    if not os.path.exists("data"):
        os.makedirs("data")

    # Step 1: Check for 512-dim dataset
    if not os.path.exists(DATA_PATH_512):
        # Step 2: Ensure original dataset exists
        if not os.path.exists(ORIGINAL_PATH):
            download_file(DATA_URL, ORIGINAL_PATH)

        # Step 3: Regenerate 512-dim embeddings
        print("[INFO] Regenerating 512-dim embeddings...")
        regenerate_embeddings(input_path=ORIGINAL_PATH, output_path=DATA_PATH_512)

    # Step 4: Load 512-dim dataset
    df = pd.read_pickle(DATA_PATH_512)
    return df
