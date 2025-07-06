from flask import Flask, request, jsonify, send_from_directory
import os
import pandas as pd
import numpy as np
from flask_cors import CORS


app = Flask(__name__, static_folder="../frontend", static_url_path='')
CORS(app)


@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

# Load and prepare data
def load_data():
    rainfall_df = pd.read_csv('data/rainfall.csv')
    normal_df = pd.read_csv('data/normal_rainfall.csv')

    # Clean column values row-wise (do NOT use .unique())
    rainfall_df['SUBDIVISION'] = rainfall_df['SUBDIVISION'].str.lower().str.strip()
    normal_df['SUBDIVISION'] = normal_df['SUBDIVISION'].str.lower().str.strip()

    return rainfall_df, normal_df


rainfall_df, normal_df = load_data()

# Prediction logic
def predict_rainfall_for_state(state, year, rainfall_df, normal_df):
    state_data = rainfall_df[rainfall_df['SUBDIVISION'] == state]

    if state_data.empty:
        return None, None, None, [], [], None

    state_data = state_data.sort_values(by='YEAR')
    history_years = state_data['YEAR'].tolist()
    history_values = state_data['ANNUAL'].tolist()

    # Simple prediction: average of last 5 years before target year
    recent_data = state_data[state_data['YEAR'] < year].tail(5)
    if recent_data.empty:
        predicted = np.mean(history_values)
    else:
        predicted = recent_data['ANNUAL'].mean()

    normal = normal_df[normal_df['SUBDIVISION'] == state]['ANNUAL RAINFALL'].values
    normal_val = normal[0] if len(normal) > 0 else predicted

    # #  Calculate % deviation
    # deviation_percent = ((predicted - normal_val) / normal_val) * 100

    # #  Risk assessment using % deviation
    # if deviation_percent > 3:
    #     risk = "Flood Risk"
    # elif deviation_percent < -20:
    #     risk = "Drought Risk"
    # else:
    #     risk = "Normal Risk"


        # Risk assessment using ratio
    ratio = predicted / normal_val
    
    if ratio >= 1.0:
        risk = "Flood Risk"
    elif ratio <= 0.9:
        risk = "Drought Risk"
    else:
        risk = "Normal Risk"

    deviation_percent = (ratio - 1) * 100  # Optional, if you want to display


    return predicted, deviation_percent, risk, history_years, history_values, normal_val


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received from frontend:", data)

    try:
        state = data['state'].strip().lower()
        year = int(data['year'])

        result = predict_rainfall_for_state(state, year, rainfall_df, normal_df)
        print("Prediction result:", result)

        if result[0] == 0 and result[2] in ["Model not found", "Insufficient data"]:
            return jsonify({"error": result[2]}), 404

        pred, deviation, risk, history_years, history_values, normal = result

        return jsonify({
            "predicted_rainfall": float(pred),
            "deviation": float(deviation),
            "risk": str(risk),
            "history_years": [int(y) for y in history_years],
            "history_rainfall": [float(v) for v in history_values],
            "normal": float(normal)
        })
    except Exception as e:
        print(" Backend error:", str(e))  # <-- will show error in terminal
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
