from flask import Flask, render_template, request,jsonify
import torch
import io
import torch.nn as nn
import torchvision
from app.torch_utils import transform_image, get_prediction
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
app = Flask(__name__)



@app.route('/')
def home():
    return "Flask App is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']

        # Check if the file is present
        if file is None or file.filename == '':
            return jsonify({'error': 'No file provided'}), 400
        
        # Check if the file is a valid image
        if not file or not file.filename.endswith(('png', 'jpg', 'jpeg')):
            return jsonify({'error': 'Invalid image format'}), 400
        try:
        # Read the image and apply transformations
            image_bytes = file.read()
            tensor = transform_image(image_bytes)
            prediction = get_prediction(tensor)

            data={'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

