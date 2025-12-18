from preprocess import load_and_preprocess
from model import build_model
import numpy as np


def train():
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "data/raw/creditcard.csv"
    )

    model = build_model(X_train.shape[1])

    # Handle class imbalance using class weights
    fraud_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
    class_weights = {0: 1.0, 1: fraud_ratio}

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    model.save("fraud_model.h5")

    return model, X_test, y_test


if __name__ == "__main__":
    train()
