from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from model import NeuralNet
import random
import cv2
import numpy as np
import json

app = Flask(__name__)
CORS(app)

from nltk_utils import tokenize, bag_of_words, stem
import torch

np.set_printoptions(suppress=True)
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Load intents from the file
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Define your NeuralNet model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


model2 = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def process_image(image):
    try:
        # Resize the image
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        # Convert to float32 and reshape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image
        image = (image / 127.5) - 1
        return image
    except Exception as e:
        return str(e)

# Function to get the model's response
def get_model_response(user_message):
    sentence = tokenize(user_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # Get the response associated with the predicted tag
    for intent in intents['intents']:
        if tag == intent["tag"]:
            return random.choice(intent['responses'])

# Flask endpoint to handle chat
@app.route('/chat', methods=['POST'])
def chatFunction():
    data = request.get_json()
    user_message = data['message']

    # Get model's response
    bot_response = get_model_response(user_message)

    return jsonify({'response': bot_response})

# Flask endpoint to check server status
@app.route('/status', methods=['GET'])
def serverStatus():
    return jsonify({'status': 'online'})

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return "No file uploaded"
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Read the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Process the image
        processed_image = process_image(image)
        if isinstance(processed_image, str):
            return processed_image

        # Predict the model
        prediction = model2.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result_message = f"Class: {class_name[2:]} Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
        return jsonify({'response': result_message})
    except Exception as e:
        return jsonify({'response': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
