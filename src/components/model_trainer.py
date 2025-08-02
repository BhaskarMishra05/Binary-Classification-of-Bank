import os
import sys
import joblib
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj, load_obj, evalution_metrics
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)

@dataclass
class MODEL_TRAINER_CONFIG:
    model_path: str = os.path.join('artifacts','model.pkl')

class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config = MODEL_TRAINER_CONFIG()

    def training_initalizer(self, train_array, test_array):
        try:
            logging.info('MODEL TRAINER STAGE STARTS >>>>')
            logging.info('Spliting the dataset into training and test set')
            X_train, y_train, X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info('Applying Synthetic Minority Oversampling Technique')
            sm= SMOTE()
            X_ref, y_ref= sm.fit_resample(X_train, y_train)
            logging.info('Creating X_ref and y_ref as containing resampled values of train and test')
            logging.info('Creating a dict or both models and params for optimal model selection and hyperparameter tuning')
            models = {
                        'LogisticRegression': LogisticRegression(random_state=42),
                        'KNeighborsClassifier': KNeighborsClassifier(),
                    }
            params={
                'LogisticRegression':{
                    'penalty':['l1','l2'],
                    'solver': ['liblinear'],
                    'max_iter': [100, 200],
                    'fit_intercept':[True,False],
                    'class_weight':['balanced']
                },
                'KNeighborsClassifier':{
                    'n_neighbors':[2,5,7],
                    'weights':['uniform','distance'],
                    'leaf_size':[3,5,8]
                }

            }
            logging.info('Using evalution metrics to find the accuarcy of each model')
            accuracy, trained_model = evalution_metrics(X_ref, y_ref, X_test, y_test, models, params)
            logging.info('Sort and find the best accuracy score')
            best_accuracy= max(sorted(accuracy.values()))
            logging.info(f'Best accuracy {best_accuracy}')
            best_model_name= list(accuracy.keys())[
                list(accuracy.values()).index(best_accuracy)
            ]
            best_model= trained_model[best_model_name]
            save_obj(
                file_path= self.model_trainer_config.model_path,
                obj= best_model
            )
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            print("Accuracy      :", accuracy_score(y_test, y_pred))
            print("Precision     :", precision_score(y_test, y_pred, pos_label='yes'))
            print("Recall        :", recall_score(y_test, y_pred, pos_label='yes'))
            print("F1 Score      :", f1_score(y_test, y_pred, pos_label='yes'))
            print("ROC AUC Score :", roc_auc_score(y_test, y_prob))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            logging.info('Model trainer stage has been executed successfully')

            return acc
        except Exception as e:
            raise CustomException(e,sys)