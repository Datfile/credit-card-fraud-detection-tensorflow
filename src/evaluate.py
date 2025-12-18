import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)

from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.show()


def plot_threshold_analysis(y_true, y_probs):
    thresholds = np.linspace(0.01, 0.99, 50)
    precisions, recalls = [], []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))

    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision–Recall vs Threshold")
    plt.legend()
    plt.show()


def evaluate(threshold=0.3):
    _, X_test, _, y_test = load_and_preprocess("data/raw/creditcard.csv")
    model = load_model("fraud_model.h5")

    y_probs = model.predict(X_test).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_probs):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_probs)
    plot_precision_recall_curve(y_test, y_probs)
    plot_threshold_analysis(y_test, y_probs)


if __name__ == "__main__":
    evaluate(threshold=0.3)
