import os
import sys
import numpy as np
import numpy.random
import numpy.random._pickle
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Patch NumPy 1.x bit_generator_ctor for models saved under NumPy 2.x
def _patched_bit_generator_ctor(bit_generator_name='MT19937'):
    return numpy.random.MT19937()

numpy.random._pickle.__bit_generator_ctor = _patched_bit_generator_ctor
sys.modules['numpy.random._mt19937'] = numpy.random

app = Flask(__name__)






# Global model and scaler initializations
ensemble_model = None
ensemble_scaler = None
lstm_model = None
lstm_scaler = None
model_load_error = None

# Load models and scalers on startup
try:
    ensemble_model = joblib.load('nifty50_ensemble_model.joblib')
    ensemble_scaler = joblib.load('ensemble_scaler.joblib')
    lstm_model = load_model('nifty50_lstm_model.keras', compile=False)
    lstm_scaler = joblib.load('lstm_scaler.joblib')
    print("Models loaded successfully!")
except Exception as e:
    model_load_error = str(e)
    print(f"Error loading models: {e}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    models_ready = all([
        ensemble_model is not None,
        ensemble_scaler is not None,
        lstm_model is not None,
        lstm_scaler is not None
    ])
    status_code = 200 if models_ready else 500
    return jsonify({
        'status': 'healthy' if models_ready else 'degraded',
        'models_loaded': models_ready,
        'error': model_load_error
    }), status_code


@app.route('/predict', methods=['POST'])
def predict():
    if not all([ensemble_model, ensemble_scaler, lstm_model, lstm_scaler]):
        return jsonify({'error': f'Models are not loaded properly on the server: {model_load_error}'}), 500


    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload provided'}), 400

        if 'ensemble_features' not in data or 'lstm_sequence' not in data:
            return jsonify({'error': 'Missing required fields: ensemble_features or lstm_sequence'}), 400
        
        # Ensemble prediction
        ens_features = np.array(data['ensemble_features']).reshape(1, -1)
        ens_scaled = ensemble_scaler.transform(ens_features)
        ens_pred = int(ensemble_model.predict(ens_scaled)[0])
        ens_conf = float(np.max(ensemble_model.predict_proba(ens_scaled)))

        # LSTM prediction
        lstm_seq = np.array(data['lstm_sequence']).reshape(1, 15, 9)
        lstm_reshaped = lstm_seq.reshape(-1, 9)
        lstm_scaled = lstm_scaler.transform(lstm_reshaped).reshape(1, 15, 9)
        
        lstm_prob = float(lstm_model.predict(lstm_scaled)[0][0])
        lstm_pred = 1 if lstm_prob > 0.5 else 0
        lstm_conf = lstm_prob if lstm_prob > 0.5 else 1 - lstm_prob

        return jsonify({
            'ensemble': {'prediction': 'UP' if ens_pred == 1 else 'DOWN', 'confidence': ens_conf},
            'lstm': {'prediction': 'UP' if lstm_pred == 1 else 'DOWN', 'confidence': lstm_conf}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

