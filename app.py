from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- LOADING SECTION ---
base_path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_path, 'crop_model.pkl'))
encoder = joblib.load(os.path.join(base_path, 'label_encoder.pkl'))
# ADD THIS LINE: Load your scaler
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl')) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture inputs
        data = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        # 2. TRANSFORM the data (This is the missing step!)
        # The model needs the data in the same "scale" it was trained on
        data_scaled = scaler.transform(np.array([data]))

        # 3. Predict using the SCALED data
        prediction_num = model.predict(data_scaled)
        crop_name = encoder.inverse_transform(prediction_num)[0]

        return jsonify({'prediction': f"{str(crop_name).capitalize()}"})

    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)