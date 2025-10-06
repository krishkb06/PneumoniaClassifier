import streamlit as st
from PIL import Image

# Title
st.title("Chest X-ray Classifier Demo")

# File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("L")
    image_resized = image.resize((224, 224))

    st.image(image_resized, caption="Uploaded X-ray", use_column_width=True)

    # Placeholder for prediction logic
    st.write("✅ Image successfully uploaded. Model prediction will go here.")
