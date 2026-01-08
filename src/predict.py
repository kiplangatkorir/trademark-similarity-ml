"""Inference: clean predictions for trademark similarity."""

import joblib
import argparse
import numpy as np
from src.data_prep import clean_text
from src.features import compute_features


def predict(name_1, name_2, model_path='models/trademark_similarity_model.pkl', 
            vectorizer_path='models/trademark_similarity_vectorizer.pkl'):
    """
    Predict similarity between two names using the trained model and vectorizer.
    
    Args:
        name_1: First trademark name
        name_2: Second trademark name
        model_path: Path to trained model
        vectorizer_path: Path to saved vectorizer
    
    Returns:
        decision: Boolean (True = confusingly similar)
        confidence: Probability score (0-1)
        features: Feature vector used for prediction
    """
    # Load model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Clean inputs
    name_1_clean = clean_text(name_1)
    name_2_clean = clean_text(name_2)
    
    # Create name pair array
    name_pair = np.array([[name_1_clean, name_2_clean]])
    
    # Compute features using the saved vectorizer
    features, _ = compute_features(name_pair, fit_vectorizer=vectorizer)
    
    # Predict
    decision = bool(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = proba[1]  # Probability of positive class (similar)
    
    return decision, confidence, features


def main():
    parser = argparse.ArgumentParser(description='Predict trademark similarity')
    parser.add_argument('name_1', type=str, help='First name')
    parser.add_argument('name_2', type=str, help='Second name')
    parser.add_argument('--model', type=str, default='models/trademark_similarity_model.pkl',
                        help='Path to trained model')
    parser.add_argument('--vectorizer', type=str, default='models/trademark_similarity_vectorizer.pkl',
                        help='Path to saved vectorizer')
    args = parser.parse_args()
    
    try:
        decision, confidence, features = predict(args.name_1, args.name_2, 
                                                  model_path=args.model,
                                                  vectorizer_path=args.vectorizer)
        
        print(f"\n=== Prediction ===")
        print(f"Name 1: {args.name_1}")
        print(f"Name 2: {args.name_2}")
        print(f"Similar (Decision): {'YES' if decision else 'NO'}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Feature vector shape: {features.shape}")
        print()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python -m src.train")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
