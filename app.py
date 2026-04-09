from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- LOADING SECTION ---
base_path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_path, 'crop_model.pkl'))
encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture inputs
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        # Predict and Decode
        prediction_num = model.predict(np.array([data]))
        crop_name = encoder.inverse_transform(prediction_num)[0]

        return jsonify({'prediction': f"{str(crop_name).capitalize()}"})

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(debug=True)