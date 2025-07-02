import pandas as pd

# -------- Rainfall Time Series Data --------
df_rain = pd.read_csv("rainfall in india 1901-2015.csv")
df_rain['Annual'] = df_rain[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].sum(axis=1)
df_rain = df_rain[['SUBDIVISION', 'YEAR', 'Annual']]
df_rain.rename(columns={"Annual": "ANNUAL"}, inplace=True)
df_rain['SUBDIVISION'] = df_rain['SUBDIVISION'].str.lower()
df_rain.to_csv("data/rainfall.csv", index=False)
print(" Saved: data/rainfall.csv")

# -------- Normal Rainfall (State-wise) --------
df_normal = pd.read_csv("district wise rainfall normal.csv")

# Group by state and average the district-level annual normal values
df_normal = df_normal.groupby('STATE_UT_NAME')['ANNUAL'].mean().reset_index()
df_normal.columns = ['SUBDIVISION', 'ANNUAL RAINFALL']
df_normal['SUBDIVISION'] = df_normal['SUBDIVISION'].str.lower()

df_normal.to_csv("data/normal_rainfall.csv", index=False)
print("Saved: data/normal_rainfall.csv")
