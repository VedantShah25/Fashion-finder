import base64
import io
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_query_embedding(text_query=None, image_path=None):
    """
    Encode multimodal query (text + image) into an embedding.
    Always returns 512-dim embeddings to match dataset.
    """

    inputs = {}
    if text_query:
        inputs["text"] = text_query
    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs["images"] = image

    processed = clip_processor(**inputs, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**processed)

    # Extract embeddings
    text_emb = None
    img_emb = None

    if text_query:
        text_emb = outputs.text_embeds.detach().cpu().numpy()[0]  # 512-dim
    if image_path:
        img_emb = outputs.image_embeds.detach().cpu().numpy()[0]  # 512-dim

    # Combine
    if text_emb is not None and img_emb is not None:
        embedding = (text_emb + img_emb) / 2.0  # average → still 512-dim
    elif text_emb is not None:
        embedding = text_emb
    elif img_emb is not None:
        embedding = img_emb
    else:
        raise ValueError("At least one of text_query or image_path must be provided.")

    return embedding
