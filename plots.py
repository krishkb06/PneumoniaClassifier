# plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig

# ROC Curve
def plot_roc(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc = roc_auc_score(y_true, y_pred_probs)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return fig

# Precision-Recall Curve
def plot_precision_recall(y_true, y_pred_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"AP = {ap:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return fig

# Sample Predictions Grid
def plot_sample_predictions(images, labels, model, class_names, n=9):
    fig, axes = plt.subplots(3, 3, figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        if i >= n: break
        img = images[i].numpy().astype("uint8")
        true_label = class_names[int(labels[i])]
        prob = model.predict(np.expand_dims(images[i],0))[0][0]
        pred_label = class_names[int(prob>0.5)]
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(f"T:{true_label}\nP:{pred_label} ({prob:.2f})")
        ax.axis("off")
    plt.tight_layout()
    return fig
