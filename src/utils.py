import os
import pandas as pd
import pickle
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
def save_obj(file_path: str, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open (file_path, 'wb') as f:
        return pickle.dump(obj, f)

def load_obj(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'rb')  as f:
        return pickle.load(f)
    

def evalution_metrics(X_train,y_train,X_test,y_test, models, params):
    try:
        accuracy={}
        model_name={}

        for name, model in models.items():
            gcv= GridSearchCV(model, param_grid=params[name], cv=5)
            gcv.fit(X_train, y_train)
            best_model = gcv.best_estimator_

            y_pred = best_model.predict(X_test)
            accuracy[name]= accuracy_score(y_test, y_pred)

            model_name[name] = best_model
        return dict(accuracy), dict(model_name)
    except Exception as e:
        raise CustomException(e,sys)