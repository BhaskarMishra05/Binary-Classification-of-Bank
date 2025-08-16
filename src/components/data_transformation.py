import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np
from dataclasses import dataclass
import joblib
from src.utils import save_obj

@dataclass
class DATA_TRANSFORMATION_CONFIG:
    proprocessed_file_path: str = os.path.join('artifacts', 'preprocessed.pkl')

class DATA_TRANSFORMATION:
    def __init__(self):
        self.data_transformation_config = DATA_TRANSFORMATION_CONFIG()

    def feature_engineering(self, train, test):
        oe_education = OrdinalEncoder(categories=[['secondary', 'primary', 'tertiary', 'unknown']])
        train['education'] = oe_education.fit_transform(train[['education']])
        test['education'] = oe_education.transform(test[['education']])

        oe_job = OrdinalEncoder(categories=[[
            'services', 'blue-collar', 'technician', 'admin.', 'housemaid',
            'entrepreneur', 'management', 'unemployed', 'self-employed',
            'student', 'retired', 'unknown'
        ]])
        train['job'] = oe_job.fit_transform(train[['job']])
        test['job'] = oe_job.transform(test[['job']])

        for col in ['default', 'loan', 'housing']:
            train[col] = train[col].map({'yes': 1, 'no': 0})
            test[col] = test[col].map({'yes': 1, 'no': 0})

        month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        train['month'] = train['month'].map(month_map)
        test['month'] = test['month'].map(month_map)

        for df in [train, test]:
            df['cyclic_sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
            df['cyclic_cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
            df['cyclic_sin_day'] = np.sin(2 * np.pi * df['day'] / 31)
            df['cyclic_cos_day'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dept'] = df['balance'].apply(lambda x: 1 if x >= 0 else 0)
            df.drop(columns=['month', 'day'], inplace=True)

        return train, test

    def transformation(self, df: pd.DataFrame):
        try:
            logging.info('DATA TRANSFORMATION STAGE STARTS >>>>')
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                                 ('scaler', StandardScaler())])
            categorical_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')),
                                                   ('encoder', OneHotEncoder(drop= 'first',
                                                                             sparse_output=False,
                                                                             handle_unknown='ignore'))])
            preprocessing = ColumnTransformer(transformers=[('num_pipeline', numerical_pipeline, numerical_columns),
                                                            ('cat_pipeline', categorical_pipeline, categorical_columns)])
            return preprocessing
        except Exception as e:
            raise CustomException(e,sys)
        
    def tranformation_initilizer(self, train_data, test_data):
        try:
            train = pd.read_csv(train_data, sep=',')
            test = pd.read_csv(test_data, sep= ',')
            train, test = self.feature_engineering(train, test)
            target = 'y'
            feature_train = train.drop(columns=[target], axis=1)
            target_train = train[target]
            feature_test = test.drop(columns=[target], axis=1)
            target_test = test[target]
            cat_col = feature_train.select_dtypes(include=['object']).columns.tolist()
            cat_index = [feature_train.columns.get_loc(col) for col in cat_col]
            smote = SMOTENC(categorical_features=cat_index, sampling_strategy='minority')
            feature_train_resample, target_train_resample= smote.fit_resample(feature_train, target_train)
            preprocessing_obj = self.transformation(feature_train_resample)
            train_transformed = preprocessing_obj.fit_transform(feature_train_resample)
            test_transformed = preprocessing_obj.transform(feature_test)
            train_arr = np.c_[train_transformed, target_train_resample]
            test_arr = np.c_[test_transformed, target_test]
            save_obj(self.data_transformation_config.proprocessed_file_path, preprocessing_obj)
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)
