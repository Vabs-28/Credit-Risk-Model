from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model AND the scaler
model = pickle.load(open('eps_v1.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result="")

@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from the form
    v1 = float(request.form['ROCE (%)']) 
    v2 = float(request.form['CASA (%)']) 
    v3 = float(request.form['Return on Equity / Networth (%)']) 
    v4 = float(request.form['Non-Interest Income/Total Assets (%)']) 
    v5 = float(request.form['Operating Profit/Total Assets (%)']) 
    v6 = float(request.form['Operating Expenses/Total Assets (%)']) 
    v7 = float(request.form['Interest Expenses/Total Assets (%)']) 
    v8 = float(request.form['Face_value']) 
    
    # Create a 2D numpy array of the raw features
    raw_features = np.array([[v1, v2, v3, v4, v5, v6, v7, v8]])
    
    # CRITICAL FIX: Scale the inputs using the saved scaler
    scaled_features = scaler.transform(raw_features)
    
    # Predict using the scaled features
    prediction = model.predict(scaled_features)[0]
    result = round(prediction, 2)

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)