# Trademark Similarity ML

A machine learning model to detect confusingly similar trademark names.

## Overview

This project trains a classifier to identify pairs of names that are confusingly similar for trademark purposes. The model uses character n-gram TF-IDF features with Logistic Regression or Gradient Boosting for classification.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare data
python -m src.data_prep

# Train model
python -m src.train

# Evaluate
python -m src.evaluate

# Make predictions
python -m src.predict "ApplePay" "Apple Pay"

# Batch predictions from CSV
python -m src.batch_predict input.csv output.csv
```

## Project Structure

- **data/**: Raw and processed data
- **src/**: Modular pipeline (data prep → features → training → evaluation → inference)
- **models/**: Trained model artifact
- **notebooks/**: Exploratory analysis

## Data Format

Input CSV (`data/raw/name_pairs.csv`):
```
name_1,name_2,label
ApplePay,Apple Pay,1
Google,Alphabet,0
```

Labels: `1 = confusingly similar`, `0 = not similar`

## Model

- **Features**: 
  - TF-IDF cosine similarity (character n-grams 2-3)
  - Edit distance (normalized Levenshtein)
  - Jaccard similarity (2-grams and 3-grams)
  - Sequence ratio (difflib)
  - Length ratio and difference
  - Common prefix ratio
- **Algorithm**: Logistic Regression with class balancing
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

## Usage

### Single Prediction

```python
from src.predict import predict

decision, confidence, features = predict("ApplePay", "Apple Pay")
print(f"Similar: {decision}, Confidence: {confidence:.3f}")
```

### Command Line

```bash
# Single prediction
python -m src.predict "ApplePay" "Apple Pay"

# Batch predictions from CSV
python -m src.batch_predict data/input.csv data/output.csv
```

The batch prediction script expects a CSV with `name_1` and `name_2` columns and outputs predictions with confidence scores.

## Features

### Enhanced Feature Engineering

The model now uses 8 features:
1. **TF-IDF Cosine Similarity**: Character n-gram (2-3) based similarity
2. **Edit Distance**: Normalized Levenshtein distance
3. **Jaccard Similarity (2-grams)**: Set-based similarity on character 2-grams
4. **Jaccard Similarity (3-grams)**: Set-based similarity on character 3-grams
5. **Sequence Ratio**: difflib-based sequence matching
6. **Length Ratio**: Ratio of minimum to maximum length
7. **Prefix Ratio**: Ratio of common prefix length
8. **Length Difference**: Absolute difference in name lengths

### Model Artifacts

- `models/trademark_similarity_model.pkl`: Trained classifier
- `models/trademark_similarity_vectorizer.pkl`: Fitted TF-IDF vectorizer (required for inference)
