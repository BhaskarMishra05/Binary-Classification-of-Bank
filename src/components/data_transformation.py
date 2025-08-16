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
        return train, test

    def transformation(self, df: pd.DataFrame):
        try:
            numerical_columns = ['age','balance','day','duration','campaign','pdays','previous']
            low_card_cat = ['marital','default','housing','loan','poutcome']
            numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
            low_card_pipeline = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                          ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])
            preprocessing = ColumnTransformer([
                ('num', numerical_pipeline, numerical_columns),
                ('low_card', low_card_pipeline, low_card_cat)
            ], remainder='passthrough')
            return preprocessing
        except Exception as e:
            raise CustomException(e, sys)

    def tranformation_initilizer(self, train_data, test_data):
        try:
            train = pd.read_csv(train_data)
            test = pd.read_csv(test_data)
            train, test = self.feature_engineering(train, test)
            target = 'y'
            feature_train = train.drop(columns=[target])
            target_train = train[target]
            feature_test = test.drop(columns=[target])
            target_test = test[target]
            categorical_cols = ['job','marital','education','default','housing','loan','contact','poutcome']
            for col in categorical_cols:
                feature_train[col] = feature_train[col].astype('category').cat.codes
                feature_test[col] = feature_test[col].astype('category').cat.codes
            cat_index = [feature_train.columns.get_loc(col) for col in categorical_cols]
            smote = SMOTENC(categorical_features=cat_index, sampling_strategy='minority')
            feature_train_resample, target_train_resample = smote.fit_resample(feature_train, target_train)
            preprocessing_obj = self.transformation(feature_train_resample)
            train_transformed = preprocessing_obj.fit_transform(feature_train_resample)
            test_transformed = preprocessing_obj.transform(feature_test)
            train_arr = np.c_[train_transformed, target_train_resample]
            test_arr = np.c_[test_transformed, target_test]
            save_obj(self.data_transformation_config.proprocessed_file_path, preprocessing_obj)
            return train_arr, test_arr
        except Exception as e:
            raise CustomException(e, sys)
