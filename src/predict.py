"""Inference: clean predictions for trademark similarity."""

import joblib
import argparse
from src.data_prep import clean_text
from src.features import compute_features


def predict(name_1, name_2, model_path='models/trademark_similarity_model.pkl', vectorizer_path=None):
    """
    Predict similarity between two names.
    
    Args:
        name_1: First trademark name
        name_2: Second trademark name
        model_path: Path to trained model
        vectorizer_path: Path to saved vectorizer (optional)
    
    Returns:
        score: Similarity score (0-1)
        decision: Boolean (True = confusingly similar)
    """
    # Load model
    model = joblib.load(model_path)
    
    # Clean inputs
    name_1_clean = clean_text(name_1)
    name_2_clean = clean_text(name_2)
    
    # Compute similarity feature
    # Note: In production, would need to save and load the vectorizer
    # For now, create a minimal feature (cosine similarity of cleaned names)
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    vec.fit([name_1_clean, name_2_clean])
    
    v1 = vec.transform([name_1_clean])
    v2 = vec.transform([name_2_clean])
    score = cosine_similarity(v1, v2)[0, 0]
    
    # Predict
    X = np.array([[score]])
    decision = bool(model.predict(X)[0])
    prob = model.predict_proba(X)[0]
    
    return score, decision, prob[1]


def main():
    parser = argparse.ArgumentParser(description='Predict trademark similarity')
    parser.add_argument('name_1', type=str, help='First name')
    parser.add_argument('name_2', type=str, help='Second name')
    args = parser.parse_args()
    
    score, decision, confidence = predict(args.name_1, args.name_2)
    
    print(f"\n=== Prediction ===")
    print(f"Name 1: {args.name_1}")
    print(f"Name 2: {args.name_2}")
    print(f"Similarity Score: {score:.3f}")
    print(f"Similar (Decision): {decision}")
    print(f"Confidence: {confidence:.3f}")
    print()


if __name__ == '__main__':
    main()
