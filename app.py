import os
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_modelv1.keras")

model = load_model()

@st.cache_resource
def load_test_ds():
    return tf.keras.preprocessing.image_dataset_from_directory(
        "chest_xray/test",
        labels="inferred",
        label_mode="binary",
        image_size=(224, 224),
        color_mode="grayscale",
        batch_size=32,
        shuffle=False
    )


try:
    test_ds = load_test_ds()

    # Get true labels
    y_true = np.concatenate([y for _, y in test_ds], axis=0)

    # Get predicted probabilities
    y_pred_probs = model.predict(test_ds)

    # Convert to binary predictions
    y_pred = (y_pred_probs > 0.5).astype("int32")

    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(y_true, y_pred, ["Normal", "Pneumonia"]))

    st.subheader("ROC Curve")
    st.pyplot(plot_roc(y_true, y_pred_probs))

    st.subheader("Precision-Recall Curve")
    st.pyplot(plot_precision_recall(y_true, y_pred_probs))

    #Sample predictions grid
    for images, labels in test_ds.take(1):
        st.subheader("Sample Predictions")
        st.pyplot(plot_sample_predictions(images, labels, model, ["Normal", "Pneumonia"]))

except FileNotFoundError:
    st.warning("Test dataset not found. Place it under chest_xray/test/")
except Exception as e:
    st.error(f"Error during evaluation: {e}")
    
# Title
st.title("Chest X-ray Classifier Demo")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("L")
    image_resized = image.resize((224, 224))
    st.image(image_resized, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess for model
    img_array = np.array(image_resized) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)

    # Run prediction
    prob = model.predict(img_array)[0][0]
    label = "Pneumonia" if prob > 0.5 else "Normal"

    # Display results
    st.success(f"🧠 Prediction: **{label}**")
    st.write(f"📊 Confidence: **{prob:.2f}**")