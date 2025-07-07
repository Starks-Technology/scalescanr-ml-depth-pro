from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import torch
from depth.models.midas_net import MidasNet
from depth.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as T

app = Flask(__name__)

# Load model
model = MidasNet("weights/depthpro.pt")
model.eval()

transform = T.Compose([
    Resize(640, 640, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal", image_interpolation_method="bicubic"),
    NormalizeImage(mean=[0.5], std=[0.5]),
    PrepareForNet()
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert("RGB")
    input_tensor = transform({"image": np.array(image)})["image"]
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)

    with torch.no_grad():
        prediction = model.forward(input_tensor)[0]
        depth = prediction.squeeze().numpy().tolist()

    return jsonify({'depth': depth})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
