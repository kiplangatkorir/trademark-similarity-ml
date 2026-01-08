"""Model training with class imbalance handling."""

from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib

from src.data_prep import load_and_prepare
from src.features import compute_features


def train_model(X_train, y_train, model_path='models/trademark_similarity_model.pkl'):
    """
    Train Logistic Regression with class weight balancing.
    
    Args:
        X_train: Feature matrix (n, num_features)
        y_train: Labels (n,)
        model_path: Where to save the model
    
    Returns:
        model: Trained classifier
    """
    # Compute class weights to handle imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {c: w for c, w in zip(classes, weights)}
    
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model


if __name__ == '__main__':
    # Full pipeline: prepare data, compute features, train
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare('data/raw/name_pairs.csv')
    
    print("Computing features...")
    features_train, vectorizer = compute_features(X_train)
    
    print("Training model...")
    model = train_model(features_train, y_train)
    
    print("Training complete.")
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Class weights used: balanced (to handle imbalance)")
