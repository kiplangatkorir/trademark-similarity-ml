"""Model evaluation: precision, recall, F1, confusion matrix."""

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import joblib

from src.data_prep import load_and_prepare
from src.features import compute_features


def evaluate_model(model, X_test, y_test):
    """
    Compute key evaluation metrics.
    
    Args:
        model: Trained classifier
        X_test: Feature matrix for test set
        y_test: True labels
    
    Returns:
        metrics: Dict with precision, recall, f1, cm
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics, y_pred


def print_metrics(metrics):
    """Pretty-print evaluation results."""
    print("\n=== Model Evaluation ===")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")


if __name__ == '__main__':
    # Full evaluation pipeline
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare('data/raw/name_pairs.csv')
    
    print("Computing features...")
    features_train, vectorizer = compute_features(X_train)
    features_test, _ = compute_features(X_test, fit_vectorizer=vectorizer)
    
    print("Loading trained model...")
    model = joblib.load('models/trademark_similarity_model.pkl')
    
    print("Evaluating...")
    metrics, y_pred = evaluate_model(model, features_test, y_test)
    print_metrics(metrics)
