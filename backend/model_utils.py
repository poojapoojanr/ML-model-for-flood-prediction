import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def load_data():
    rainfall_df = pd.read_csv("data/rainfall.csv")
    normal_df = pd.read_csv("data/normal_rainfall.csv")

    # Standardize state names to lowercase
    rainfall_df['SUBDIVISION'] = rainfall_df['SUBDIVISION'].str.lower()
    normal_df['SUBDIVISION'] = normal_df['SUBDIVISION'].str.lower()

    return rainfall_df, normal_df

def predict_rainfall_for_state(state, year, rainfall_df, normal_df):
    # Historical data for the state
    df_state = rainfall_df[rainfall_df['SUBDIVISION'] == state].sort_values('YEAR')
    if len(df_state) < 10:
        return 0, 0, "Insufficient data", [], [], 0

    # Use last 5 years of annual rainfall
    last_5 = df_state['ANNUAL'].values[-5:].reshape(-1, 1)

    # Normalize for LSTM
    scaler = MinMaxScaler()
    last_5_scaled = scaler.fit_transform(last_5).reshape(1, 5, 1)

    # Load the LSTM model
    model_path = f"models/model_{state.replace(' ', '_')}.h5"
    if not os.path.exists(model_path):
        return 0, 0, "Model not found", [], [], 0
    model = load_model(model_path, compile=False)

    # Predict
    predicted_scaled = model.predict(last_5_scaled)
    predicted_rainfall = scaler.inverse_transform(predicted_scaled)[0][0]

    # Get normal rainfall for comparison
    normal_row = normal_df[normal_df['SUBDIVISION'] == state]
    normal_rainfall = normal_row['ANNUAL RAINFALL'].values[0] if not normal_row.empty else 0

    deviation = ((predicted_rainfall - normal_rainfall) / normal_rainfall) * 100 if normal_rainfall > 0 else 0

    # Risk categorization
    if deviation > 30:
        risk = "Flood Risk"
    elif deviation < -20:
        risk = "Drought Risk"
    else:
        risk = "Normal"

    # For charting
    history_years = df_state['YEAR'].values[-10:].tolist()
    history_values = df_state['ANNUAL'].values[-10:].tolist()

    return predicted_rainfall, deviation, risk, history_years, history_values, normal_rainfall
