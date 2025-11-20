from flask import Flask, request, jsonify
from src.models.io import load_model
from src.data.loader import load_dataset
import numpy as np

app = Flask(__name__)
model = None

@app.route('/load', methods=['POST'])
def load():
    global model
    path = request.json.get("path")
    model = load_model(path)
    return {"status": "loaded"}

@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.json["data"]
    arr = np.array(data).reshape(1, -1)
    pred = model.predict(arr)[0]
    return {"prediction": int(pred)}

