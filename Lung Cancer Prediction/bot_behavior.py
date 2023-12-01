from flask import Flask, request, render_template
import pickle
import numpy as np
import json
import requests
import pandas as pd



app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/inspect")
def predict():
    return render_template("inspect.html")


@app.route('/output',methods=["POST","GET"])# route to show the predictions in a web UI
def submit():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]  
    #input_feature = np.transpose(input_feature)
    x=[np.array(input_feature)]
    print(input_feature)
    names = ['index','Patient Id' ,'Age', 'Gender', 'Air Pollution', 'Alcohol use','Dust Allergy', 'OccuPational Hazards', 'Genetic Risk','chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking','Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue','Weight Loss', 'Shortness of Breath', 'Wheezing','Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold','Dry Cough','Snoring']
    data = pd.DataFrame(x,columns=names)
    print(data)
    pred = model.predict(data)
    if(pred == 1):
        return render_template('output.html', predict="Yes")
    else:
        return render_template('output.html', predict="No")

if __name__ == "__main__":
    
    app.run(debug = True,port = 4444)