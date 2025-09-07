import streamlit as st
import base64
from io import BytesIO
from PIL import Image

from utils.dataset_loader import load_dataset
from utils.preprocessing import get_query_embedding
from models.retriever import Retriever
from models.generator import Generator


# -----------------------------
# Helper: Convert image to base64
# -----------------------------
def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# -----------------------------
# Main Streamlit App
# -----------------------------
st.set_page_config(page_title="Fashion Finder", layout="wide")
st.title("Fashion Finder (Multimodal RAG)")
st.write("Upload an image and/or enter a text query to discover similar fashion styles!")

# Load dataset once
@st.cache_resource
def load_resources():
    df = load_dataset()
    retriever = Retriever(df)
    generator = Generator(model="llama3")
    return df, retriever, generator

df, retriever, generator = load_resources()

# -----------------------------
# Input Section
# -----------------------------
text_query = st.text_input("Enter a text query (optional):")
uploaded_file = st.file_uploader("Upload an image query (optional):", type=["jpg", "jpeg", "png"])

query_image_path = None
if uploaded_file:
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Uploaded Image", use_column_width=True)
    # Save temporarily for CLIP processing
    query_image_path = "temp_query.png"
    query_image.save(query_image_path)

# Run Search
if st.button("Search"):
    if not text_query and not query_image_path:
        st.warning("Please provide either a text query, an image, or both.")
    else:
        # 1. Get query embedding
        query_embedding = get_query_embedding(text_query=text_query, image_path=query_image_path)

        # 2. Retrieve similar items
        retrieved_items = retriever.retrieve(query_embedding, top_k=5)

        # 3. Show results
        st.subheader("Similar Products")
        cols = st.columns(5)
        for i, (_, row) in enumerate(retrieved_items.iterrows()):
            try:
                img = row["Image"]
                if isinstance(img, bytes):  # if images are stored as raw bytes
                    img = Image.open(BytesIO(img)).convert("RGB")
                elif isinstance(img, str):  # if image paths are stored
                    img = Image.open(img).convert("RGB")

                img_b64 = image_to_base64(img)
                img_tag = f'<img src="data:image/png;base64,{img_b64}" width="200"><br>{row["Item Name"]}<br>Price: {row["Price"]}'
                cols[i].markdown(img_tag, unsafe_allow_html=True)
            except Exception as e:
                cols[i].write(f"{row['Item Name']} (Image unavailable)")

        # 4. Generate response
        st.subheader("Fashion Assistant Suggestion")
        response = generator.generate(text_query if text_query else "User uploaded an image", retrieved_items)
        st.write(response)
