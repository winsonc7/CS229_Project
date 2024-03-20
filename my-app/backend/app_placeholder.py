from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)

# Load your TensorFlow model
model = tf.keras.models.load_model('models/neural_jbench_100.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    input_data = [float(request.form['input1']), float(request.form['input2']), ...]

    # Preprocess the input data as needed for your model
    # Example: Convert input_data to a NumPy array and normalize/scale it

    # Make predictions using your TensorFlow model
    predictions = model.predict([input_data])

    # Format the predictions as needed
    # Example: Convert predictions to a human-readable format

    # Pass the predictions to the HTML template for display
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
