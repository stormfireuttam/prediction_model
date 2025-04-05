from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    value = float(data['value'])
    prediction = model.predict(np.array([[value]]))
    return jsonify({'predicition' : prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)