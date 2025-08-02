import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            path_model = 'artifacts/model.pkl'
            path_preprocessing = 'artifacts/preprocessed.pkl'
            
            logging.info(f"Loading model from {path_model}")
            model = load_obj(path_model)
            
            logging.info(f"Loading preprocessor from {path_preprocessing}")
            preprocessing = load_obj(path_preprocessing)

            logging.info(f"Input features:\n{features}")
            data_loader = preprocessing.transform(features)
            logging.info(f"Transformed data shape: {data_loader.shape}")

            preds = model.predict(data_loader)
            logging.info(f"Predictions: {preds}")
            
            return preds

        except Exception as e:
            import traceback
            logging.error("Prediction Failed!")
            logging.error(traceback.format_exc())
            raise CustomException(e, sys)

class CUSTOMDATA:
    def __init__(self, age, job, marital, education, default, balance,
                 housing, loan, contact, day, month, duration,
                 campaign, pdays, previous, poutcome):
        self.age = age
        self.job = job
        self.marital = marital
        self.education = education
        self.default = default
        self.balance = balance
        self.housing = housing
        self.loan = loan
        self.contact = contact
        self.day = day
        self.month = month
        self.duration = duration
        self.campaign = campaign
        self.pdays = pdays
        self.previous = previous 
        self.poutcome = poutcome

    def data_to_dataframe(self):
        try:
            dict_of_data = {
                'age': self.age,
                'job': self.job,
                'marital': self.marital,
                'education': self.education,
                'default': self.default,
                'balance': self.balance,
                'housing': self.housing,
                'loan': self.loan,
                'contact': self.contact,
                'day': self.day,
                'month': self.month,
                'duration': self.duration,
                'campaign': self.campaign,
                'pdays': self.pdays,
                'previous': self.previous,
                'poutcome': self.poutcome
            }
            return pd.DataFrame([dict_of_data])
        except Exception as e:
            raise CustomException(e, sys)
