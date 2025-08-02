import os
import sys
from src.pipeline.prediction_pipeline import CUSTOMDATA, PredictionPipeline
from flask import render_template, Flask, request
from src.logger import logging 
from src.exception import CustomException

application = Flask(__name__)
app = application


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET','POST'])

def predict_datapoints():
    if request.method=="GET":
        return render_template('home.html')
    
    else:
        data = CUSTOMDATA(
                age = request.form.get('age'),
                job = request.form.get('job'),
                marital = request.form.get('marital'),
                education = request.form.get('education'),
                default = request.form.get('default'),
                balance = request.form.get('balance'),
                housing = request.form.get('housing'),
                loan = request.form.get('loan'),
                contact = request.form.get('contact'),
                day = request.form.get('day'),
                month = request.form.get('month'),
                duration = request.form.get('duration'),
                campaign = request.form.get('campaign'),
                pdays = request.form.get('pdays'),
                previous = request.form.get('previous'),
                poutcome = request.form.get('poutcome')
                )
        pred_data = data.data_to_dataframe()
        print(pred_data)
        pred_pipeline = PredictionPipeline()
        pred=  pred_pipeline.predict(pred_data)
        result = "Yes" if pred[0] == 1 else "No"
        return render_template('home.html', results= result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
