"""Feature engineering: n-gram TF-IDF and similarity."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_features(name_pairs, max_features=500, ngram_range=(2, 3), fit_vectorizer=None):
    """
    Compute character n-gram TF-IDF features and cosine similarity.
    
    Args:
        name_pairs: Array of shape (n, 2) with [name_1, name_2]
        max_features: Max number of n-gram features
        ngram_range: (min_n, max_n) for character n-grams
        fit_vectorizer: Pre-fitted vectorizer (for test set), or None to fit
    
    Returns:
        features: Feature matrix of shape (n, num_features)
        vectorizer: Fitted vectorizer (for reuse on test set)
    """
    if fit_vectorizer is None:
        vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=False  # Already cleaned in data_prep
        )
        # Fit on all names in the pairs
        all_names = np.concatenate([name_pairs[:, 0], name_pairs[:, 1]])
        vectorizer.fit(all_names)
    else:
        vectorizer = fit_vectorizer
    
    # Transform each name
    name1_vec = vectorizer.transform(name_pairs[:, 0])
    name2_vec = vectorizer.transform(name_pairs[:, 1])
    
    # Compute cosine similarity for each pair
    similarities = np.array([
        cosine_similarity(name1_vec[i], name2_vec[i]).item()
        for i in range(len(name_pairs))
    ]).reshape(-1, 1)
    
    return similarities, vectorizer


if __name__ == '__main__':
    # Test on example
    from data_prep import load_and_prepare
    X_train, X_test, _, _ = load_and_prepare('data/raw/name_pairs.csv')
    features, vec = compute_features(X_train)
    print(f"Feature shape: {features.shape}")
    print(f"Sample similarities: {features[:5].flatten()}")
