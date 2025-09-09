# app.py
import streamlit as st
from PIL import Image
import io
import base64

from utils.dataset_loader import load_dataset
from utils.preprocessing import get_query_embedding
from models.retriever import Retriever
from models.generator import Generator

# -------------------------
# Load resources once
# -------------------------
@st.cache_data(show_spinner=True)
def load_resources():
    df = load_dataset()
    retriever = Retriever(df)
    generator = Generator()
    return df, retriever, generator

df, retriever, generator = load_resources()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Fashion Finder", layout="wide")
st.title("Fashion Finder")

# Input query
text_query = st.text_input("Enter fashion query:")
image_file = st.file_uploader("Upload an outfit image:", type=["jpg", "jpeg", "png"])

query_image_path = None

if image_file:
    query_image_path=f"temp_{image_file.name}"
    with open(query_image_path, "wb") as f:
        f.write(image_file.getbuffer())
    st.image(Image.open(query_image_path), caption="Uploaded Image", use_container_width=True)

# Submit button
if st.button("Search"):
    if not text_query and not image_file:
        st.warning("Please provide a text query and/or an image!")
    else:
        # Convert uploaded image to temporary file-like object
        image_path = None
        if image_file:
            image_bytes = image_file.read()
            image_path = io.BytesIO(image_bytes)

        # -------------------------
        # Generate query embedding
        # -------------------------
        query_embedding = get_query_embedding(text_query=text_query, image_path=image_path)

        # -------------------------
        # Retrieve similar items
        # -------------------------
        retrieved_items = retriever.retrieve(query_embedding, top_k=5)

        # -------------------------
        # Generate AI response
        # -------------------------
        response_text = generator.generate(text_query, retrieved_items)

        # -------------------------
        # Display results
        # -------------------------
        st.subheader("AI Response:")
        st.write(response_text)

        st.subheader("Top Similar Fashion Items:")
        for idx, row in retrieved_items.iterrows():
            st.markdown(f"**{row['Item Name']}**")
            st.markdown(f"Price: {row['Price']} | [Buy Here]({row['Link']})")
            # Inline Base64 image
            if "Encoded Image" in row and row["Encoded Image"]:
                img_bytes = base64.b64decode(row["Encoded Image"])
                image = Image.open(io.BytesIO(img_bytes))
                st.image(image, use_container_width=True)
            st.markdown("---")
