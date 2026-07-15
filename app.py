
from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and scalers
try:
    ensemble_model = joblib.load('nifty50_ensemble_model.joblib')
    ensemble_scaler = joblib.load('ensemble_scaler.joblib')
    lstm_model = load_model('nifty50_lstm_model.keras')
    lstm_scaler = joblib.load('lstm_scaler.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Expecting a flat list of features for Ensemble
        # features: ['RSI_14', 'Return', 'MACD_diff', 'Price_Dist_SMA', 'Lag_Ret_1', 'Lag_Ret_2', 'Lag_Ret_3']
        ens_features = np.array(data['ensemble_features']).reshape(1, -1)
        ens_scaled = ensemble_scaler.transform(ens_features)
        ens_pred = int(ensemble_model.predict(ens_scaled)[0])
        ens_conf = float(np.max(ensemble_model.predict_proba(ens_scaled)))

        # Expecting a sequence (window=15) for LSTM
        # shape: (1, 15, 9)
        lstm_seq = np.array(data['lstm_sequence']).reshape(1, 15, 9)
        # Note: Scaler usually applied to 2D then reshaped to 3D
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
    app.run(port=5000)
