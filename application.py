from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/predictdata', methods=['POST', 'GET'])
def predictdata():
    if request.method == 'POST':
        # Get all features from form
        temperature = request.form.get('Temperature')
        rh = request.form.get('RH')
        ws = request.form.get('Ws')
        rain = request.form.get('Rain')
        ffmc = request.form.get('FFMC')
        dmc = request.form.get('DMC')
        dc = request.form.get('DC')
        isi = request.form.get('ISI')
        bui = request.form.get('BUI')
        classes = request.form.get('Classes')
        region = request.form.get('Region')
        
        # Convert numeric values to float for processing
        numeric_data = [float(temperature), float(rh), float(ws), float(rain), 
                       float(ffmc), float(dmc), float(dc), float(isi), float(bui)]
        
        # Scale the data and make prediction
        data_scaled = scaler.transform([numeric_data])
        prediction = ridge_model.predict(data_scaled)
        return render_template('home.html', result=prediction[0])
    else:
        return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)