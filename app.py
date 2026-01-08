"""Flask web application for trademark similarity predictions."""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from src.predict import predict

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Health check endpoint
@app.route('/')
def index():
    """Root endpoint - serve HTML interface or return API information."""
    # Check if HTML template exists
    try:
        return render_template('index.html')
    except:
        # Fallback to JSON if template not found
        return jsonify({
            'message': 'Trademark Similarity ML API',
            'version': '1.0',
            'endpoints': {
                '/': 'API information',
                '/health': 'Health check',
                '/predict': 'Predict similarity (POST)',
                '/predict/batch': 'Batch predictions (POST)'
            }
        })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        # Try to load the model to verify it exists
        import joblib
        model_path = os.getenv('MODEL_PATH', 'models/trademark_similarity_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'models/trademark_similarity_vectorizer.pkl')
        
        joblib.load(model_path)
        joblib.load(vectorizer_path)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict_similarity():
    """
    Predict similarity between two trademark names.
    
    Expected JSON body:
    {
        "name_1": "ApplePay",
        "name_2": "Apple Pay"
    }
    
    Returns:
    {
        "name_1": "ApplePay",
        "name_2": "Apple Pay",
        "is_similar": true,
        "confidence": 0.804,
        "prediction": 1
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        name_1 = data.get('name_1')
        name_2 = data.get('name_2')
        
        if not name_1 or not name_2:
            return jsonify({
                'error': 'Missing required fields: name_1 and name_2 are required'
            }), 400
        
        # Get model paths from environment or use defaults
        model_path = os.getenv('MODEL_PATH', 'models/trademark_similarity_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'models/trademark_similarity_vectorizer.pkl')
        
        # Make prediction
        decision, confidence, features = predict(
            name_1, 
            name_2,
            model_path=model_path,
            vectorizer_path=vectorizer_path
        )
        
        return jsonify({
            'name_1': name_1,
            'name_2': name_2,
            'is_similar': decision,
            'confidence': float(confidence),
            'prediction': 1 if decision else 0
        }), 200
        
    except FileNotFoundError as e:
        return jsonify({
            'error': 'Model files not found. Please ensure the model is trained and model files exist.',
            'details': str(e)
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Expected JSON body:
    {
        "pairs": [
            {"name_1": "ApplePay", "name_2": "Apple Pay"},
            {"name_1": "Google", "name_2": "Alphabet"}
        ]
    }
    
    Returns:
    {
        "results": [
            {
                "name_1": "ApplePay",
                "name_2": "Apple Pay",
                "is_similar": true,
                "confidence": 0.804,
                "prediction": 1
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        pairs = data.get('pairs', [])
        
        if not pairs or not isinstance(pairs, list):
            return jsonify({
                'error': 'Missing or invalid "pairs" field. Expected a list of objects with name_1 and name_2'
            }), 400
        
        # Get model paths from environment or use defaults
        model_path = os.getenv('MODEL_PATH', 'models/trademark_similarity_model.pkl')
        vectorizer_path = os.getenv('VECTORIZER_PATH', 'models/trademark_similarity_vectorizer.pkl')
        
        results = []
        for pair in pairs:
            name_1 = pair.get('name_1')
            name_2 = pair.get('name_2')
            
            if not name_1 or not name_2:
                results.append({
                    'name_1': name_1,
                    'name_2': name_2,
                    'error': 'Missing name_1 or name_2'
                })
                continue
            
            try:
                decision, confidence, _ = predict(
                    name_1,
                    name_2,
                    model_path=model_path,
                    vectorizer_path=vectorizer_path
                )
                
                results.append({
                    'name_1': name_1,
                    'name_2': name_2,
                    'is_similar': decision,
                    'confidence': float(confidence),
                    'prediction': 1 if decision else 0
                })
            except Exception as e:
                results.append({
                    'name_1': name_1,
                    'name_2': name_2,
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Batch prediction failed',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

