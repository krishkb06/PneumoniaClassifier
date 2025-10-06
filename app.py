import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

from plots import (
    plot_confusion_matrix,
    plot_roc,
    plot_precision_recall,
    plot_sample_predictions
)

model = tf.keras.models.load_model("pneumonia_model.keras")

# Sample ground truth labels (0 = Normal, 1 = Pneumonia)
y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])

# Sample predicted probabilities
y_pred_probs = np.array([0.10, 0.85, 0.40, 0.95, 0.20, 0.70, 0.80, 0.30, 0.65, 0.15])

# Thresholded predictions (using 0.5 cutoff)
y_pred = (y_pred_probs > 0.5).astype("int32")

st.subheader("Confusion Matrix")
st.pyplot(plot_confusion_matrix(y_true, y_pred, ["Normal", "Pneumonia"]))

st.subheader("ROC Curve")
st.pyplot(plot_roc(y_true, y_pred_probs))

st.subheader("Precision-Recall Curve")
st.pyplot(plot_precision_recall(y_true, y_pred_probs))

#not displaying this one right now as it requires my model and dataset which are not uploaded currently
"""
for images, labels in test_ds.take(1):
    st.subheader("Sample Predictions")
    st.pyplot(plot_sample_predictions(images, labels, model, ["Normal", "Pneumonia"]))
"""
    
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
