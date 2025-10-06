# Chest X-Ray Pneumonia Classifier

A TensorFlow/Keras convolutional neural network (CNN) that classifies grayscale chest X-ray images as NORMAL or PNEUMONIA. This repository demonstrates an end-to-end machine learning workflow — data preprocessing, model training, evaluation, interpretability (Grad-CAM), and deployment via a Streamlit web app.

## Project overview

- Task: Binary image classification on chest X-rays (NORMAL vs PNEUMONIA).
- Frameworks: TensorFlow / Keras for model development; Streamlit for lightweight deployment and demo.
- Input: Grayscale chest X-ray images resized to a consistent resolution and normalized for training.
- Output: Probability/confidence score and class label (NORMAL or PNEUMONIA). Optional Grad-CAM overlay for interpretability.

Real-world impact: Early and accurate detection of pneumonia from X-rays can support clinicians by prioritizing cases, reducing diagnostic latency, and improving patient outcomes. This project explores how applied AI can augment clinical workflows while emphasizing interpretability and careful evaluation.

## Why I built this

As someone who lives with asthma and has personally experienced pneumonia and lung-related complications, I wanted to explore how machine learning can help in medical imaging. This project is a professional, research-focused effort to learn and demonstrate practical ML techniques for healthcare applications while maintaining an emphasis on interpretability and responsible use.

## Dataset

This project uses the Kaggle Chest X-Ray Pneumonia dataset (commonly shared as the Chest X-Ray Images (Pneumonia) dataset). The training pipeline expects the following folder layout:

- train/
  - NORMAL/
  - PNEUMONIA/
- val/
  - NORMAL/
  - PNEUMONIA/
- test/
  - NORMAL/
  - PNEUMONIA/

Each subfolder contains grayscale JPEG/PNG X-ray images labeled by folder. The code performs on-the-fly augmentation for training and deterministic preprocessing for validation/test.

If you don't have the dataset, download it from Kaggle and extract it into the repository (or update paths in the training script).

## Model architecture

High-level summary of the CNN used in training:

- Repeated Conv block: Conv2D -> BatchNormalization -> ReLU -> MaxPooling
- Several convolutional blocks (increasing filters: e.g., 32 -> 64 -> 128)
- GlobalAveragePooling2D to reduce spatial dimensions
- Dense layers with dropout for regularization
- Final Dense(1) with sigmoid activation for binary classification

This design focuses on a compact, robust architecture suited for grayscale medical images with strong regularization and batch normalization to stabilize training.

## Evaluation & interpretability

Model performance is measured and visualized using:

- Metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix (visual)
- ROC curve and AUC
- Precision-Recall curve
- Sample predictions showing true vs predicted labels and confidence scores
- Grad-CAM heatmaps: visual explanations highlighting image regions that influenced the model's decision

These artifacts are generated during evaluation and can be viewed in notebooks or the demo app.

## Deployment

The trained model can be served via the included Streamlit app (`app.py`). Features:

- Upload a chest X-ray image and receive a prediction (NORMAL or PNEUMONIA) with a confidence score.
- Optional Grad-CAM overlay to visualize the model's focus areas.

Run locally with Streamlit for a quick demo or containerize the app for production deployment.

## Setup & quick start

1. Clone the repository

	git clone https://github.com/krishkb06/PneumoniaClassifier.git
	cd PneumoniaClassifier-1

2. Create and activate a virtual environment (Windows PowerShell example)

	python -m venv my_env; .\my_env\Scripts\Activate.ps1

3. Install dependencies

	pip install -r requirements.txt

4. Run the Streamlit app

	streamlit run app.py

Notes:

- Ensure the dataset is placed in the expected train/val/test folder structure or adjust dataset paths in the training scripts.
- For reproducible results, use the provided `requirements.txt` and the virtual environment.

## Screenshots (placeholders)

- Training & validation loss/accuracy curves
- Confusion matrix
- ROC curve and AUC
- Streamlit demo screenshot showing upload UI and prediction with Grad-CAM overlay

Replace these with real images from your runs in `/docs` or the repository root when available.

## Key takeaways

- Demonstrates a full ML lifecycle: data preprocessing, CNN training, quantitative evaluation, interpretability (Grad-CAM), and deployment.
- Emphasizes responsible, interpretable ML for healthcare use-cases.
- Useful example for recruiters and interviewers: shows skills in TensorFlow/Keras, data pipelines, model evaluation, explainability, and simple web deployment.

## License

This project is released under the MIT License. See the LICENSE file for details.

