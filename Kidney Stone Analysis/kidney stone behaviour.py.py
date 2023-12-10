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
def Index():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == 'POST':
        var1 = request.form["GRAVITY"]
        var2 = request.form["PH"]
        var3 = request.form["OSMO"]
        var4 = request.form["COND"]
        var5 = request.form["UREA"]
        var6 = request.form["CALC"]
        var7 = request.form["TARGET"]
        
        # Convert the input data into a numpy array
        predict_data = np.array([var1, var2, var3, var4, var5, var6, var7,]).reshape(1, -1)

        # Use the loaded model to make predictions
        pred = model.predict(predict_data)
        if(pred == 1):
            return render_template('output.html', predict=" +ve")
        else:
            return render_template('output.html', predict="-ve")
           


if __name__ == "__main__":
    app.run(debug=False)
