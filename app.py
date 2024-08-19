from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io


app = Flask(__name__)
model = load_model("D:\Deep learning\proposed_model.h5")

def preprocess_image():
    file = request.files['file']
    img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check if the file is an image
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Preprocess the image
        img_array = preprocess_image()
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Convert prediction to human-readable format
        if prediction[0][0] > prediction[0][1]:
            result = 'Healthy'
        else:
            result = 'Brown spots'
            
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=True)
