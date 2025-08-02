import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

    
    def transformation(self, df: pd.DataFrame):
        try:
            logging.info('DATA TRANSFORMATION STAGE STARTS >>>>')
            logging.info('Starting data transformation stage')
            logging.info('Getting numerical and categorical columns')
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            logging.info(f'Numerical Columns: {numerical_columns}')
            logging.info(f'Categorical Columns: {categorical_columns}')
            logging.info('Making pipelines of both numerical and categorical columns')
            logging.info('Applying SimpleImputer , Stadardrization and Encoding techniques on numerical and categorical columns')
            numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
                                                 ('scaler', StandardScaler())])
            categorical_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')),
                                                   ('encoder', OneHotEncoder(drop= 'first',sparse_output=False ,handle_unknown='ignore'))])
            logging.info('Applying Column Transformer on both the pipelines')
            preprocessing = ColumnTransformer(transformers=[('num_pipeline', numerical_pipeline, numerical_columns),
                                                            ('cat_pipeline', categorical_pipeline, categorical_columns)])
            logging.info('Preprocessing Stage has been executed successfully')
            logging.info('Now we will be using this function for transforming our train and test dataset')
            return preprocessing
        except Exception as e:
            raise CustomException(e,sys)
        
    def tranformation_initilizer(self, train_data, test_data):
        try:
            logging.info('Starting Transfromation part')
            logging.info('Loading train and test dataset')
            train = pd.read_csv(train_data, sep=',')
            test = pd.read_csv(test_data, sep= ',')
            logging.info(f'{train.columns}')
            target = 'y'
            logging.info(f'Define target columns, which is :{target}')
            logging.info('Dividing the dataset based on  feature and target for both train and test')
            feature_train = train.drop(columns=[target], axis=1)
            target_train = train[target]
            feature_test = test.drop(columns=[target], axis=1)
            target_test = test[target]
            logging.info(f"Train shape: {train.shape}")
            logging.info(f"Test shape: {test.shape}")
            logging.info('Initialising preoprocessing function and passing train features into it')
            preprocessing_obj = self.transformation(feature_train)
            train_transformed = preprocessing_obj.fit_transform(feature_train)
            test_transformed = preprocessing_obj.transform(feature_test)
            logging.info('Concatening the transformed train, test with the target variable of respective sets')
            train_arr = np.c_[train_transformed, target_train]
            test_arr = np.c_[test_transformed, target_test]
            logging.info('Saving the preprocessed file into artifacts as preprocessed.pkl')
            save_obj(self.data_transformation_config.proprocessed_file_path, preprocessing_obj)
            logging.info('Succcessfully executed the Transformation stage')
            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    