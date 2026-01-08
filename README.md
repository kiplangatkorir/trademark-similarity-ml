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

## Deployment on Render

This application can be deployed on Render as a web service. The Flask API provides REST endpoints for trademark similarity predictions.

### Prerequisites

1. Ensure the model is trained locally and the model files exist in the `models/` directory
2. Commit the model files to your repository (they should NOT be in `.gitignore`)
3. Have a Render account (free tier available)

### Deployment Steps

1. **Push your code to GitHub/GitLab/Bitbucket**

2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your repository

3. **Configure the service:**
   - **Name**: trademark-similarity-api (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free tier is sufficient

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application

### API Endpoints

Once deployed, your API will be available at `https://your-app-name.onrender.com`

#### Health Check
```bash
GET /
GET /health
```

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "name_1": "ApplePay",
  "name_2": "Apple Pay"
}
```

Response:
```json
{
  "name_1": "ApplePay",
  "name_2": "Apple Pay",
  "is_similar": true,
  "confidence": 0.804,
  "prediction": 1
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "pairs": [
    {"name_1": "ApplePay", "name_2": "Apple Pay"},
    {"name_1": "Google", "name_2": "Alphabet"}
  ]
}
```

### Testing the API Locally

Before deploying, test the API locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py

# Or use gunicorn (production server)
gunicorn app:app
```

Then test with curl:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"name_1": "ApplePay", "name_2": "Apple Pay"}'
```

### Environment Variables (Optional)

You can set these in Render's dashboard if you have custom model paths:
- `MODEL_PATH`: Path to model file (default: `models/trademark_similarity_model.pkl`)
- `VECTORIZER_PATH`: Path to vectorizer file (default: `models/trademark_similarity_vectorizer.pkl`)
- `PORT`: Port number (automatically set by Render)
