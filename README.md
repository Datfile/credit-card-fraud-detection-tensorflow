# Credit Card Fraud Detection using TensorFlow

## Overview
This project detects fraudulent credit card transactions using a Deep Learning model built with TensorFlow (Keras).
Due to extreme class imbalance, evaluation focuses on recall, precision, PR-AUC, and ROC-AUC rather than accuracy alone.

## Dataset
An anonymized dataset of European cardholder transactions with highly imbalanced classes.

## Approach
1. Data preprocessing and scaling
2. Handling class imbalance using class weights
3. Neural Network modeling with TensorFlow
4. Threshold tuning
5. Model evaluation using multiple metrics

## Model
- Feedforward Neural Network
- ReLU activations
- Dropout regularization
- Adam optimizer

## Evaluation Metrics
- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy

## Tech Stack
Python, TensorFlow, Pandas, NumPy, Scikit-learn
