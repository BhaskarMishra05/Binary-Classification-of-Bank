import os
import sys
from flask import Flask, render_template, request, jsonify
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

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json()
        input_data = CustomData(
            age=int(data.get("age", 0)),
            job=data.get("job", ""),
            marital=data.get("marital", ""),
            education=data.get("education", ""),
            default=data.get("default", ""),
            balance=float(data.get("balance", 0)),
            housing=data.get("housing", ""),
            loan=data.get("loan", ""),
            contact=data.get("contact", ""),
            day=int(data.get("day", 0)),
            month=data.get("month", ""),
            duration=float(data.get("duration", 0)),
            campaign=int(data.get("campaign", 0)),
            pdays=int(data.get("pdays", 0)),
            previous=int(data.get("previous", 0)),
            poutcome=data.get("poutcome", "")
        )
        df = input_data.to_dataframe()
        pipeline = PredictionPipeline()
        pred = pipeline.predict(df)
        result = "yes" if pred[0] == 1 else "no"
        return jsonify({"prediction": result})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
