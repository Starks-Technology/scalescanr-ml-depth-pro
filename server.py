from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import torch
import os
import logging
import requests
from urllib.parse import urlparse
import traceback
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and transform
model = None
transform = None
model_loaded = False
model_error = None

def load_model():
    """Load the DepthPro model and transform synchronously."""
    global model, transform, model_loaded, model_error
    
    try:
        logger.info("Starting model loading...")
        from depth_pro import create_model_and_transforms
        
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        
        logger.info(f"Loading model on device: {device}")
        
        model, transform = create_model_and_transforms(
            device=device,
            precision=torch.float32,
        )
        
        logger.info("Model loaded successfully")
        model.eval()
        model_loaded = True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        model_error = str(e)
        model_loaded = False

def validate_image_url(url):
    """Validate that the URL is accessible and returns an image."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"
        
        # Check if URL is accessible
        response = requests.head(url, timeout=10)
        if response.status_code != 200:
            return False, f"URL returned status code {response.status_code}"
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
            return False, f"URL does not point to an image (content-type: {content_type})"
        
        return True, "URL is valid"
    except Exception as e:
        return False, f"Error validating URL: {str(e)}"

@app.route('/')
def home():
    """Home endpoint with API information."""
    return jsonify({
        "message": "DepthPro API",
        "model_status": "loaded" if model_loaded else "loading" if not model_error else "error",
        "model_error": model_error,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with image_url in JSON)"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint."""
    if model_loaded:
        return jsonify({"status": "healthy", "model": "loaded"})
    elif model_error:
        return jsonify({"status": "error", "model": "error", "error": model_error}), 500
    else:
        return jsonify({"status": "loading", "model": "loading"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict depth map from image URL."""
    if not model_loaded:
        if model_error:
            return jsonify({"error": f"Model failed to load: {model_error}"}), 500
        else:
            return jsonify({"error": "Model is still loading, please try again in a moment"}), 503
    
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({"error": "Missing image_url in request body"}), 400
        
        image_url = data['image_url']
        logger.info(f"Processing image from URL: {image_url}")
        
        # Download image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Load image as PIL Image
        image = Image.open(io.BytesIO(response.content))
        logger.info(f"Downloaded image: {image.size} {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to model's expected size (384x384)
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
        logger.info(f"Resized image to: {image.size}")
        
        # Apply transform to PIL Image
        input_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            depth_map = prediction.cpu().numpy().squeeze()
        
        logger.info(f"Generated depth map: {depth_map.shape}")
        
        # Convert depth map to list for JSON serialization
        depth_list = depth_map.tolist()
        
        return jsonify({
            "success": True,
            "depth_map": depth_list,
            "image_size": [384, 384],  # Model always processes 384x384
            "depth_map_shape": list(depth_map.shape)
        })
        
    except requests.RequestException as e:
        logger.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model during startup
    load_model()
    
    # Get port from environment variable (for Cloud Run)
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
