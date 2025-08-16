import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj
from sklearn.preprocessing import OrdinalEncoder

class PredictionPipeline:
    def __init__(self):
        self.model_path = 'artifacts/model.pkl'
        self.preprocessor_path = 'artifacts/preprocessed.pkl'

        self.oe_education = OrdinalEncoder(categories=[['secondary','primary','tertiary','unknown']])
        self.oe_job = OrdinalEncoder(categories=[[
            'services', 'blue-collar', 'technician', 'admin.', 'housemaid',
            'entrepreneur', 'management', 'unemployed', 'self-employed',
            'student', 'retired', 'unknown'
        ]])
   
        self.oe_default = OrdinalEncoder(categories=[['no','yes','unknown']])
        self.oe_housing = OrdinalEncoder(categories=[['no','yes','unknown']])
        self.oe_loan = OrdinalEncoder(categories=[['no','yes','unknown']])
        self.oe_marital = OrdinalEncoder(categories=[['single','married','divorced','unknown']])
        self.oe_contact = OrdinalEncoder(categories=[['cellular','telephone','unknown']])
        self.oe_poutcome = OrdinalEncoder(categories=[['unknown','failure','other','success']])

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        month_map = {'jan': 1,'feb': 2,'mar': 3,'apr': 4,'may': 5,'jun': 6,
                     'jul': 7,'aug': 8,'sep': 9,'oct': 10,'nov': 11,'dec': 12}
        df['month'] = df.get('month', pd.Series([1]*len(df))).map(month_map).fillna(1)
        df['day'] = df.get('day', pd.Series([15]*len(df))).fillna(15)

        df['cyclic_sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cyclic_cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['cyclic_sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
        df['cyclic_cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dept'] = pd.to_numeric(df['balance'], errors='coerce').apply(lambda x: 1 if x >= 0 else 0)

        if 'education' in df.columns: df['education'] = self.oe_education.fit_transform(df[['education']])
        if 'job' in df.columns: df['job'] = self.oe_job.fit_transform(df[['job']])
        if 'default' in df.columns: df['default'] = self.oe_default.fit_transform(df[['default']])
        if 'housing' in df.columns: df['housing'] = self.oe_housing.fit_transform(df[['housing']])
        if 'loan' in df.columns: df['loan'] = self.oe_loan.fit_transform(df[['loan']])
        if 'marital' in df.columns: df['marital'] = self.oe_marital.fit_transform(df[['marital']])
        if 'contact' in df.columns: df['contact'] = self.oe_contact.fit_transform(df[['contact']])
        if 'poutcome' in df.columns: df['poutcome'] = self.oe_poutcome.fit_transform(df[['poutcome']])

        return df

    def predict(self, features: pd.DataFrame):
        try:
            features = features.copy()
            numeric_cols = ['age','balance','day','duration','campaign','pdays','previous']
            for col in numeric_cols:
                features[col] = pd.to_numeric(features.get(col, 0), errors='coerce').fillna(0)

            features = self.feature_engineering(features)

            preprocessor = load_obj(self.preprocessor_path)
            model = load_obj(self.model_path)

            missing_cols = set(preprocessor.feature_names_in_) - set(features.columns)
            for col in missing_cols:
                features[col] = 0
            features = features[preprocessor.feature_names_in_]

            data_transformed = preprocessor.transform(features)
            preds = model.predict(data_transformed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
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

    def to_dataframe(self):
        data = {
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
        return pd.DataFrame([data])
