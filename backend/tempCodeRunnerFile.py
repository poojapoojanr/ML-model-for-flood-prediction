import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("data/rainfall.csv")
df['SUBDIVISION'] = df['SUBDIVISION'].str.lower()
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

states = df['SUBDIVISION'].unique()
for state in states:
    df_state = df[df['SUBDIVISION'] == state].sort_values("YEAR")
    values = df_state['ANNUAL'].values

    if len(values) < 10:
        print(f"Skipping {state} (insufficient data)")
        continue

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(5, len(values_scaled)):
        X.append(values_scaled[i-5:i])
        y.append(values_scaled[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(5, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    filename = f"model_{state.replace(' ', '_')}.h5"
    model.save(os.path.join(output_dir, filename))
    print(f"✅ Saved: {filename}")

