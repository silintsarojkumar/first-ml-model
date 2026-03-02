import pickle
from flask import Flask,request,jsonify,render_template
import numpy  as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app = application


## Importt ridge regression model standared scaler picker
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standared_scaller = pickle.load(open('models/scaler.pkl','rb'))

## Home Page




@app.route('/')
def index():
    return render_template('index.html'),


@app.route('/predictdata',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standared_scaller.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0')
