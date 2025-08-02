import pandas as pd
import numpy as np
import sys
from src.components.data_ingestion import DATA_INGESTION
from src.components.data_transformation import DATA_TRANSFORMATION
from src.components.model_trainer import MODEL_TRAINER
from src.logger import logging
from src.exception import CustomException

if __name__ == '__main__':
    try:
        data_ingestion = DATA_INGESTION()
        raw_path, train_path, test_path = data_ingestion.data_ingestion_initializer()
    except Exception as e:
        raise CustomException(e,sys)

    try:
        data_transformation = DATA_TRANSFORMATION()
        train_arr, test_arr = data_transformation.tranformation_initilizer(train_data= train_path, test_data= test_path)
    except Exception as e:
        raise CustomException(e, sys)
    
    try:
        model_trainer = MODEL_TRAINER()
        accuracy = model_trainer.training_initalizer(train_arr, test_arr)
        
    except Exception as e:
        raise CustomException(e,sys)