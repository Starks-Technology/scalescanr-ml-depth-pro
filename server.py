from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import torch
import os
import logging
import threading
import time
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and transform
model = None
transform = None
model_loading = False
model_loaded = False

def load_model_background():
    """Load the DepthPro model and transform in the background."""
    global model, transform, model_loading, model_loaded
    
    if model_loading:
        return
    
    model_loading = True
    try:
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
        model.eval()
        model_loaded = True
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
    finally:
        model_loading = False

def download_image_from_url(image_url):
    """Download image from URL and return as PIL Image."""
    try:
        logger.info(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Open image from bytes
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        logger.error(f"Failed to download image from URL: {e}")
        raise Exception(f"Failed to download image from URL: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        if model_loading:
            return jsonify({'error': 'Model is still loading, please try again in a moment'}), 503
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if image_url is provided in JSON body
    if request.is_json:
        data = request.get_json()
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'error': 'No image_url provided in request body'}), 400
    else:
        # Check if image_url is provided as form data
        image_url = request.form.get('image_url')
        if not image_url:
            return jsonify({'error': 'No image_url provided. Send image_url in JSON body or form data'}), 400

    try:
        # Download image from URL
        pil_image = download_image_from_url(image_url)
        
        # Save temporarily for processing
        temp_path = "/tmp/downloaded_image.jpg"
        pil_image.save(temp_path)
        
        # Load image using the proper function
        from depth_pro import load_rgb
        image, _, f_px = load_rgb(temp_path)
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            prediction = model.infer(input_tensor, f_px=f_px)
            depth = prediction["depth"].detach().cpu().numpy().squeeze().tolist()
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            'depth': depth,
            'image_url': image_url,
            'image_size': {
                'width': pil_image.width,
                'height': pil_image.height
            }
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    if model_loaded:
        return jsonify({'status': 'healthy', 'model': 'loaded'})
    elif model_loading:
        return jsonify({'status': 'loading', 'model': 'loading'})
    else:
        return jsonify({'status': 'unhealthy', 'model': 'not_loaded'})

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'DepthPro API',
        'model_status': 'loaded' if model_loaded else 'loading' if model_loading else 'not_loaded',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST with image_url)'
        },
        'usage': {
            'method': 'POST',
            'body': {
                'image_url': 'https://example.com/image.jpg'
            },
            'example': {
                'curl': 'curl -X POST -H "Content-Type: application/json" -d \'{"image_url":"https://example.com/image.jpg"}\' https://ss-depth-pro-1089081045712.us-central1.run.app/predict'
            }
        }
    })

if __name__ == '__main__':
    # Start model loading in background thread
    model_thread = threading.Thread(target=load_model_background, daemon=True)
    model_thread.start()
    
    # Get port from environment variable (for Cloud Run)
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting server on port {port}")
    logger.info("Model loading in background...")
    
    app.run(host='0.0.0.0', port=port)
