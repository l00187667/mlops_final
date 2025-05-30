import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('dataset/car_volume_co2.csv')

# Cleaning
df = df.dropna()
df = df[df['Volume'] > 0]
df = df[df['CO2'] > 0]

# Prepare input and output
X = df[['Volume']]
y = df['CO2']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
