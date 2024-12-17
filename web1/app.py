from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load Model Tensorflow
model = tf.keras.models.load_model('training_efficiennet.h5')

# Route untuk Prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input 
        data = request.json
        input_data = np.array([data['input']]) # Asumsikan input dalam bentuk list

        # Lakukan prediksi
        prediction = model.predict(np.expand_dims(input_data, axis=0))
        result = prediction.tolist()

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})
    
# Endpoint testing
@app.route('/')
def home():
    return "API for Model Prediction is running!"

if __name__ == '__main__':
    app.run(debug=True)  # Jalankan aplikasi dengan mode debug
