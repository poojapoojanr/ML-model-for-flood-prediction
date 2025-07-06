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








































# import pandas as pd
# import numpy as np
# import os
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, LSTM, Dense
# from tensorflow.keras.utils import to_categorical

# # Load datasets
# df = pd.read_csv("data/rainfall.csv")
# df['SUBDIVISION'] = df['SUBDIVISION'].str.lower()

# normal_df = pd.read_csv("data/normal_rainfall.csv")
# normal_df['SUBDIVISION'] = normal_df['SUBDIVISION'].str.lower()

# # Fix column name
# normal_dict = dict(zip(normal_df['SUBDIVISION'], normal_df['ANNUAL RAINFALL']))

# # Output directory
# output_dir = "models_classification"
# os.makedirs(output_dir, exist_ok=True)

# # How many past years to use
# WINDOW_SIZE = 10

# # Process each region
# states = df['SUBDIVISION'].unique()
# for state in states:
#     df_state = df[df['SUBDIVISION'] == state].sort_values("YEAR")
#     values = df_state['ANNUAL'].values

#     if len(values) <= WINDOW_SIZE:
#         print(f"⚠️ Not enough data to build 1 sequence for: {state} (years={len(values)})")
#         continue

#     # Normalize rainfall
#     scaler = MinMaxScaler()
#     values_scaled = scaler.fit_transform(values.reshape(-1, 1))

#     X, y = [], []
#     normal_val = normal_dict.get(state, None)

#     if normal_val is None:
#         print(f"⚠️ Using fallback normal value for: {state}")
#         normal_val = np.mean(values)  # fallback

#     for i in range(WINDOW_SIZE, len(values)):
#         X.append(values_scaled[i - WINDOW_SIZE:i])
#         actual = values[i]

#         # Risk classification
#         if actual > normal_val * 1.0:
#             label = 2  # Flood
#         elif actual < normal_val * 0.8:
#             label = 0  # Drought
#         else:
#             label = 1  # Normal

#         y.append(label)

#     if len(X) == 0:
#         print(f"❌ Skipped {state} — not enough sequences.")
#         continue

#     X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
#     y = to_categorical(y, num_classes=3)

#     # Build LSTM model
#     model = Sequential()
#     model.add(Input(shape=(WINDOW_SIZE, 1)))
#     model.add(LSTM(64, activation='relu'))
#     model.add(Dense(3, activation='softmax'))  # 3 classes

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     model.fit(X, y, epochs=100, verbose=0)

#     # Save model
#     model_filename = f"model_{state.replace(' ', '_')}_classifier.h5"
#     model.save(os.path.join(output_dir, model_filename))
#     print(f"✅ Saved: {model_filename}")
