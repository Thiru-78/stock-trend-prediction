# stock-trend-prediction
# Stock Trend Prediction API

A machine learning-powered REST API built with Flask that predicts stock market trends (specifically Nifty 50). The API leverages two different modeling approaches—an Ensemble model and a Deep Learning (LSTM) model—to predict whether the stock trend will go 'UP' or 'DOWN' and provides a confidence score for each.

## Features

- **Dual Model Predictions:** Returns predictions from both an Ensemble model (using traditional technical indicators) and a Long Short-Term Memory (LSTM) neural network (using sequential data).
- **Confidence Scoring:** Outputs the probability and confidence level for each prediction.
- **Production-Ready:** Configured with `gunicorn` and a `Procfile` for easy cloud deployment.
- **Robust Error Handling:** Validates incoming JSON payloads and gracefully handles shape mismatches.

## Tech Stack

- **Language:** Python 3
- **Web Framework:** Flask
- **Machine Learning:** TensorFlow/Keras (LSTM), Scikit-Learn (Ensemble models & Scalers)
- **Model Serialization:** Joblib
- **Server:** Gunicorn

## Repository Structure

- `app.py`: The main Flask application and API routing.
- `nifty50_ensemble_model.joblib` & `nifty50_lstm_model.keras`: The trained machine learning models.
- `ensemble_scaler.joblib` & `lstm_scaler.joblib`: Scalers used to normalize input data before prediction.
- `requirements.txt`: Python dependencies required to run the app.
- `Procfile`: Command to run the application in production environments (like Heroku or Render).

## Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Thiru-78/stock-trend-prediction.git](https://github.com/Thiru-78/stock-trend-prediction.git)
   cd stock-trend-prediction

   
