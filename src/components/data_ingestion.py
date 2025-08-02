import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class DATA_INGESTION_CONFIG:
        raw_data_path: str = os.path.join('artifacts', 'raw.csv')
        train_data_path: str = os.path.join('artifacts', 'train.csv')
        test_data_path: str = os.path.join('artifacts', 'test.csv')

class DATA_INGESTION:
        def __init__ (self):
                self.data_ingestion_config = DATA_INGESTION_CONFIG()

        def data_ingestion_initializer(self):
                try:
                        
                    logging.info('DATA INGESTION STAGE STARTS >>>>')
                    logging.info('Data Ingestion stage started successfully')
                    os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
                    logging.info('Successfully created the directory for storing datasets')
                    logging.info('Reading the dataset')
                    df = pd.read_csv(self.data_ingestion_config.raw_data_path, sep=';')
                    df.columns = df.columns.str.replace('"','').str.strip()
                    logging.info(f"Cleaned columns: {df.columns.tolist()}")
                    logging.info('Splitting dataset in train and test set')
                    train_dataset, test_dataset = train_test_split(df, random_state=42, test_size= 0.2)
                    logging.info('Making the csv files of train and test datasets')
                    train_dataset.to_csv(self.data_ingestion_config.train_data_path, index= False, sep= ',')
                    test_dataset.to_csv(self.data_ingestion_config.test_data_path, index= False, sep= ',')
                    logging.info('Succefully executed the Data Ingestion stage')
                    return(
                            self.data_ingestion_config.raw_data_path,
                            self.data_ingestion_config.train_data_path,
                            self.data_ingestion_config.test_data_path
                    )
                except Exception as e:
                       raise CustomException(e,sys)
        



                

