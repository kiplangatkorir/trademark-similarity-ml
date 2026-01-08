"""Data preparation: load, clean, and split."""

import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Normalize text: lowercase, remove punctuation, strip spaces."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare(input_path, test_size=0.2, random_state=42):
    """
    Load CSV, clean text, and split into train/test.
    
    Args:
        input_path: Path to CSV with columns [name_1, name_2, label]
        test_size: Fraction for test set
        random_state: Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test (names and labels)
    """
    df = pd.read_csv(input_path)
    
    # Clean both name columns
    df['name_1'] = df['name_1'].apply(clean_text)
    df['name_2'] = df['name_2'].apply(clean_text)
    
    # Create feature pairs (will be used by features.py)
    X = df[['name_1', 'name_2']].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Test pipeline
    X_train, X_test, y_train, y_test = load_and_prepare('data/raw/name_pairs.csv')
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Label distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
