from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import json
import requests



app = Flask(__name__)
with open('lr.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route("/")
def f():
    return render_template("index.html")

@app.route("/inspect")
def inspect():
    return render_template("inspect.html")


@app.route("/output", methods=["GET", "POST"])
def output():
    if request.method == 'POST':
        var1 = float(request.form["Loan_ID"])
        var2 = float(request.form["Gender"])
        var3 = float(request.form["Married"])
        var4 = float(request.form["Dependents"])
        var5 = float(request.form["Education"])
        var6 = float(request.form["Self_Employed"])
        var7 = float(request.form["ApplicantIncome"])
        var8 = float(request.form["CoapplicantIncome"])
        var9 = float(request.form["LoanAmount"])
        var10 = float(request.form["Loan_Amount_Term"])
        var11 = float(request.form["Credit_History"])
        var12 = float(request.form["Property_Area"])

        # Convert the input data into a numpy array
        predict_data = np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12]).reshape(1, -1)

        # Use the loaded model to make predictions
        predict = model.predict(predict_data)

        if (predict == 1):
            return render_template('output.html', predict="Eligible")
        else:
            return render_template('output.html', predict="Not Eligible")
    return render_template("output.html")

if __name__ == "__main__":
    app.run(debug=True,port=5000)
