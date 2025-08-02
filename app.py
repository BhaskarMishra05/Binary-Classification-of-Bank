import os
import sys
from flask import render_template, Flask, request
from src.pipeline.prediction_pipeline import CUSTOMDATA, PredictionPipeline
from src.logger import logging 
from src.exception import CustomException
import traceback

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictions', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == "GET":
        return render_template('home.html')
    
    else:
        try:
            # Collect form inputs
            data = CUSTOMDATA(
                age=request.form.get('age'),
                job=request.form.get('job'),
                marital=request.form.get('marital'),
                education=request.form.get('education'),
                default=request.form.get('default'),
                balance=request.form.get('balance'),
                housing=request.form.get('housing'),
                loan=request.form.get('loan'),
                contact=request.form.get('contact'),
                day=request.form.get('day'),
                month=request.form.get('month'),
                duration=request.form.get('duration'),
                campaign=request.form.get('campaign'),
                pdays=request.form.get('pdays'),
                previous=request.form.get('previous'),
                poutcome=request.form.get('poutcome')
            )

            # Convert to dataframe
            pred_data = data.data_to_dataframe()
            print("=== INPUT DATAFRAME ===")
            print(pred_data)

            # Run prediction
            pred_pipeline = PredictionPipeline()
            results = pred_pipeline.predict(pred_data)
            print("=== PREDICTION RESULT ===")
            print(results)

            return render_template('home.html', results=results[0])

        except Exception as e:
            # Print full error
            print("=== ERROR OCCURRED ===")
            traceback.print_exc()

            # Optional: Log with your custom logger too
            logging.error("Prediction error", exc_info=True)

            return f"<h2 style='color:red;'>Internal Server Error: {str(e)}</h2>", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
