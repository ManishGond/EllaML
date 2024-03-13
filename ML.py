from flask import Flask, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS  # Import CORS module

app = Flask(__name__)
CORS(app)  # Add CORS support to your Flask app

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to process uploaded image
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
        prediction = model.predict(processed_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        result_message = f"Class: {class_name[2:]} Confidence Score: {str(np.round(confidence_score * 100))[:-2]}%"
        return jsonify({'response': result_message})
    except Exception as e:
        return jsonify({'response': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
