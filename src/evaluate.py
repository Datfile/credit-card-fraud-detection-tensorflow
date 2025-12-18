import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess


def evaluate(threshold=0.5):
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "data/raw/creditcard.csv"
    )

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


if __name__ == "__main__":
    evaluate(threshold=0.3)
