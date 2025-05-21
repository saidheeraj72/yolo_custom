# app.py
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pretrained model
MODEL_PATH = "model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to model input size (adjust based on your model)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def process_image(image_bytes):
    """Process the image and prepare it for the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension

def get_prediction(image_tensor):
    """Get prediction from the model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        
    # This part depends on your model's output format
    # Assuming YOLOv5 or similar format that returns detections
    if hasattr(outputs, 'xyxy'):  # YOLOv5 format
        detections = outputs.xyxy[0].cpu().numpy()  # Get detections for first image
        if len(detections) > 0:
            # Get the detection with highest confidence
            best_detection = detections[np.argmax(detections[:, 4])]
            return {
                'detected': True,
                'bbox': best_detection[:4].tolist(),  # x1, y1, x2, y2
                'confidence': float(best_detection[4])
            }
    # For models that return class probabilities directly
    elif isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        if len(probs) == 1:  # Single class model
            confidence = float(probs[0])
            if confidence > 0.5:  # Threshold can be adjusted
                return {
                    'detected': True,
                    'confidence': confidence
                }
    
    # Default response if no detection or different model format
    # You may need to adjust this based on your specific model output
    return {'detected': False}

@app.route('/detect', methods=['POST'])
def detect_muzzle():
    """API endpoint to detect cattle muzzle in an image."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image file
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Process image and get prediction
        image_tensor = process_image(image_bytes)
        result = get_prediction(image_tensor)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)