from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    volume = float(request.form['volume'])
    prediction = model.predict(np.array([[volume]]))
    return render_template('index.html', prediction_text=f'Predicted CO2 Emission: {prediction[0]:.2f} g/km')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
