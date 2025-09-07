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
    """Encode multimodal query (text + image) into an embedding."""
    inputs = {}
    if text_query:
        inputs["text"] = text_query
    if image_path:
        image = Image.open(image_path).convert("RGB")
        inputs["images"] = image
    
    processed = clip_processor(**inputs, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**processed)
    embedding = outputs.pooler_output.detach().cpu().numpy()[0]
    return embedding

