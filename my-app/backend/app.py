from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import os
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
import pandas as pd

app = Flask(__name__)
CORS(app)

project_root = os.path.dirname(os.path.abspath(__file__))

FEATURE_FILE = 'mmlu_all_data_features_500.json'
FEATURE_PATH = os.path.join(project_root, 'features', FEATURE_FILE)

RAW_DATA_FILE = 'mmlu_all_data.json'
RAW_DATA_PATH = os.path.join(project_root, 'raw_data', RAW_DATA_FILE)

DATA_VEC_FILE = 'mmlu_all_data_500.csv'
DATA_VEC_PATH = os.path.join(project_root, 'data_vectors', DATA_VEC_FILE)

MODEL_FILE = 'neural_mmlu_500.h5'
MODEL_PATH = os.path.join(project_root, 'models', MODEL_FILE)

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

def get_predictions(query, num_outputs):
    outputs = []
    with open(FEATURE_PATH, 'r') as f:
        features = json.load(f).keys()
    feature_count = {feature: 0 for feature in features}

    data_vec = pd.read_csv(DATA_VEC_PATH)
    data_vec_x = data_vec.drop(columns=['y'])
    data_vec_y = data_vec['y']

    with open(RAW_DATA_PATH, 'r') as f:
        raw_data = json.load(f)
    
    cleaned_query = clean_text(query)
    for word in cleaned_query.split():
        if word in feature_count:
            feature_count[word] += 1
    input_vec = np.array([[feature_count[feature] for feature in features]])
    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(input_vec)
    subject = np.argmax(probs, axis=1)[0]
    for i in range(len(data_vec_y)):
        if data_vec_y[i] == subject:
            similarity = 1 - cosine(data_vec_x.iloc[i].values, input_vec.flatten())
            if len(outputs) < num_outputs:
                outputs.append([i, similarity])
                outputs.sort(key=lambda x: x[1], reverse=True)
            elif similarity > outputs[-1][1]:
                outputs[-1] = [i, similarity]
                outputs.sort(key=lambda x: x[1], reverse=True)
    for i in range(len(outputs)):
        index = outputs[i][0]
        question = raw_data[index]['question']
        outputs[i] = question
    return outputs

@app.route('/api/predict', methods=['POST'])
def post_output():
    data = request.get_json()
    input_string = data.get('input_string')
    num_outputs = int(data.get('num_outputs'))

    if not input_string:
        return jsonify({'error': 'Input string is missing or empty'}), 400
    
    output_list = get_predictions(input_string, num_outputs)
    return jsonify(output_list)

if __name__ == '__main__':
    app.run(debug=True)
