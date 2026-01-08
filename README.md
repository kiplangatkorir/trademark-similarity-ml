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
python -m src.predict
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

- **Features**: Character n-gram (2-3) TF-IDF
- **Algorithm**: Logistic Regression
- **Metric**: F1 score (weighted for class imbalance)

## Usage

```python
from src.predict import predict

score, decision = predict("ApplePay", "Apple Pay")
print(f"Similarity: {score:.3f}, Similar: {decision}")
```
