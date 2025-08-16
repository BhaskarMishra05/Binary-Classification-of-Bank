import os
import sys
from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == "GET":
        return render_template('home.html')
    try:
        data = CustomData(
            age=int(request.form.get('age')),
            job=request.form.get('job'),
            marital=request.form.get('marital'),
            education=request.form.get('education'),
            default=request.form.get('default'),
            balance=float(request.form.get('balance')),
            housing=request.form.get('housing'),
            loan=request.form.get('loan'),
            contact=request.form.get('contact'),
            day=int(request.form.get('day')),
            month=request.form.get('month'),
            duration=float(request.form.get('duration')),
            campaign=int(request.form.get('campaign')),
            pdays=int(request.form.get('pdays')),
            previous=int(request.form.get('previous')),
            poutcome=request.form.get('poutcome')
        )
        pred_data = data.to_dataframe()
        pred_pipeline = PredictionPipeline()
        pred = pred_pipeline.predict(pred_data)
        result = "Yes" if pred[0] == 1 else "No"
        return render_template('home.html', results=result)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
