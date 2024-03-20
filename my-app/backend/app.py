from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def tokenize_input():
    data = request.get_json()
    input_string = data.get('input_string')

    if not input_string:
        return jsonify({'error': 'Input string is missing or empty'}), 400

    tokens = input_string.split()  # Split the input string into tokens
    num_tokens = len(tokens)

    token_list = [{'token': input_string} for token in tokens]  # Create a list of tokens

    response_data = {
        'input_string': input_string,
        'tokens': token_list,
        'num_tokens': num_tokens
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
