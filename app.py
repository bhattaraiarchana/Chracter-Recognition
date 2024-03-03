from flask import Flask, render_template, request
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model architecture from JSON file
model_path = r"D:\Nepali Handwritten Character\ncrs\sample_model\model.json"
with open(model_path, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights(r"D:\Nepali Handwritten Character\ncrs\sample_model\model (1).h5")

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']
    
    # Load and preprocess the image
    img = Image.open(img_file)
    resized_image = img.resize((32, 32))
    grayscale_image = resized_image.convert('L')
    img_array = img_to_array(grayscale_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1

    # Make prediction using the model
    prediction = loaded_model.predict(img_array)

    # Convert prediction to class label
    predicted_class = np.argmax(prediction)

    # Define class mapping
    class_mapping = {
        0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ', 5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ', 10: 'ट',
        11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण', 15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न', 20: 'प',
        21: 'फ', 22: 'ब', 23: 'भ', 24: 'म', 25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श', 30: 'ष',
        31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ', 36: '०', 37: '१', 38: '२', 39: '३',
        40: '४', 41: '५', 42: '६', 43: '७', 44: '८', 45: '९'
    }

    # Get the predicted character
    predicted_character = class_mapping[predicted_class]

    # Convert image array back to image for visualization
    img_array *= 255.0  # Reverse normalization
    img_array = img_array.astype(np.uint8)  # Convert to uint8
    image = Image.fromarray(img_array[0, :, :, 0], 'L')  # Convert to PIL Image

    # Save the processed image
    processed_image_path = 'static/processed_image.png'
    image.save(processed_image_path)

    return render_template('result.html', prediction=predicted_character, processed_image=processed_image_path)

if __name__ == '__main__':
    app.run(debug=True)
