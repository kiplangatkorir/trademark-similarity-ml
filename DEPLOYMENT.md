# Deployment Guide for Render

This guide walks you through deploying the Trademark Similarity ML API on Render.

## Prerequisites

1. ✅ Trained model files in `models/` directory:
   - `models/trademark_similarity_model.pkl`
   - `models/trademark_similarity_vectorizer.pkl`

2. ✅ Git repository with all code committed

3. ✅ Render account (sign up at [render.com](https://render.com) - free tier available)

## Important: Commit Model Files

**Before deploying, ensure your model files are committed to Git:**

```bash
# Check if model files are tracked
git status models/

# If they're not tracked, add them
git add models/*.pkl
git commit -m "Add trained model files for deployment"
git push
```

**Note:** Model files (`.pkl`) are typically large. If they exceed GitHub's file size limits (100MB), consider using Git LFS or storing them externally and downloading during build.

## Deployment Steps

### Option 1: Using Render Dashboard (Recommended)

1. **Push your code to GitHub/GitLab/Bitbucket**
   ```bash
   git push origin main
   ```

2. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Sign in or create an account

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your repository (authorize Render to access your repo)

4. **Configure Service Settings**
   - **Name**: `trademark-similarity-api` (or your preferred name)
   - **Region**: Choose closest to you
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: (leave empty)
   - **Environment**: `Python 3` (will use Python 3.12 from runtime.txt)
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

5. **Optional: Set Environment Variables**
   - Click "Advanced" → "Add Environment Variable"
   - Usually not needed (uses defaults)
   - If you have custom paths:
     - `MODEL_PATH`: Path to model file
     - `VECTORIZER_PATH`: Path to vectorizer file

6. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (typically 2-5 minutes)
   - Your API will be available at `https://your-app-name.onrender.com`

### Option 2: Using render.yaml (Infrastructure as Code)

If you prefer to define your service in code, use the provided `render.yaml`:

1. **Ensure render.yaml is committed**
   ```bash
   git add render.yaml
   git commit -m "Add Render configuration"
   git push
   ```

2. **In Render Dashboard**
   - Click "New +" → "Blueprint"
   - Select your repository
   - Render will automatically detect and use `render.yaml`

## Testing Your Deployment

Once deployed, test your API:

### Health Check
```bash
curl https://your-app-name.onrender.com/health
```

### Single Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"name_1": "ApplePay", "name_2": "Apple Pay"}'
```

### Batch Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {"name_1": "ApplePay", "name_2": "Apple Pay"},
      {"name_1": "Google", "name_2": "Alphabet"}
    ]
  }'
```

## Testing Locally First

Before deploying, test the API locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Or with gunicorn (production-like)
gunicorn app:app
```

Then test:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"name_1": "ApplePay", "name_2": "Apple Pay"}'
```

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### Model Files Not Found
- Ensure model files are committed to Git
- Check file paths are correct
- Verify files exist in `models/` directory

### API Returns 500 Error
- Check application logs in Render dashboard
- Verify model files are present and loadable
- Test locally first to identify issues

### Service Spins Down (Free Tier)
- Free tier services spin down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds (cold start)
- Consider upgrading to paid tier for always-on service

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

See README.md for detailed API documentation.

## Cost Considerations

- **Free Tier**: 
  - Service spins down after inactivity
  - Limited resources
  - Good for testing and low-traffic use
  
- **Paid Tier**: 
  - Always-on service
  - Better performance
  - Starts at $7/month

Choose based on your usage needs!

