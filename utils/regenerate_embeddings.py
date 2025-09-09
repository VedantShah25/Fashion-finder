# utils/regenerate_embeddings.py
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def regenerate_embeddings(input_path: str, output_path: str):
    """
    Loads original dataset, computes 512-dim CLIP embeddings for images,
    and saves new dataset to output_path.
    """
    df = pd.read_pickle(input_path)
    embeddings = []

    for img_data in tqdm(df["Encoded Image"], desc="Generating embeddings"):
        # Convert base64 to PIL Image
        img_bytes = io.BytesIO(base64.b64decode(img_data))
        image = Image.open(img_bytes).convert("RGB")

        # Compute embedding
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        outputs = clip_model.get_image_features(**inputs)
        emb = outputs.detach().cpu().numpy()[0]
        embeddings.append(emb)

    df["Embedding"] = embeddings
    df.to_pickle(output_path)
    print(f"[INFO] 512-dim embeddings saved to {output_path}")
