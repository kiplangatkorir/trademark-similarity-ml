"""Feature engineering: n-gram TF-IDF and multiple similarity metrics."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib


def _edit_distance(s1, s2):
    """Compute normalized Levenshtein edit distance (0-1 scale, 1=identical)."""
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    # Create distance matrix
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    max_len = max(len(s1), len(s2))
    return 1.0 - (dp[m][n] / max_len)


def _jaccard_similarity(s1, s2, n=2):
    """Compute Jaccard similarity on character n-grams."""
    def get_ngrams(s, n):
        return set([s[i:i+n] for i in range(len(s)-n+1)])
    
    set1 = get_ngrams(s1, n)
    set2 = get_ngrams(s2, n)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    return intersection / union


def _sequence_ratio(s1, s2):
    """Compute sequence matching ratio using difflib."""
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_additional_features(name_pairs):
    """
    Compute additional similarity features beyond TF-IDF cosine.
    
    Args:
        name_pairs: Array of shape (n, 2) with [name_1, name_2]
    
    Returns:
        additional_features: Array of shape (n, num_additional_features)
    """
    features = []
    
    for name1, name2 in name_pairs:
        # Edit distance (normalized Levenshtein)
        edit_sim = _edit_distance(name1, name2)
        
        # Jaccard similarity on 2-grams
        jaccard_2 = _jaccard_similarity(name1, name2, n=2)
        
        # Jaccard similarity on 3-grams
        jaccard_3 = _jaccard_similarity(name1, name2, n=3)
        
        # Sequence ratio (difflib)
        seq_ratio = _sequence_ratio(name1, name2)
        
        # Length features
        len_diff = abs(len(name1) - len(name2))
        len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2)) if max(len(name1), len(name2)) > 0 else 0
        
        # Common prefix length
        prefix_len = 0
        min_len = min(len(name1), len(name2))
        for i in range(min_len):
            if name1[i] == name2[i]:
                prefix_len += 1
            else:
                break
        prefix_ratio = prefix_len / max(len(name1), len(name2)) if max(len(name1), len(name2)) > 0 else 0
        
        features.append([
            edit_sim,
            jaccard_2,
            jaccard_3,
            seq_ratio,
            len_ratio,
            prefix_ratio,
            len_diff
        ])
    
    return np.array(features)


def compute_features(name_pairs, max_features=500, ngram_range=(2, 3), fit_vectorizer=None):
    """
    Compute comprehensive features: TF-IDF cosine + additional similarity metrics.
    
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
    tfidf_cosine = np.array([
        cosine_similarity(name1_vec[i], name2_vec[i]).item()
        for i in range(len(name_pairs))
    ]).reshape(-1, 1)
    
    # Compute additional features
    additional_features = compute_additional_features(name_pairs)
    
    # Concatenate all features
    all_features = np.hstack([tfidf_cosine, additional_features])
    
    return all_features, vectorizer


if __name__ == '__main__':
    # Test on example
    from data_prep import load_and_prepare
    X_train, X_test, _, _ = load_and_prepare('data/raw/name_pairs.csv')
    features, vec = compute_features(X_train)
    print(f"Feature shape: {features.shape}")
    print(f"Sample similarities: {features[:5].flatten()}")
