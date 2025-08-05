import os
import sys
import joblib
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass
from sklearn.metrics import (accuracy_score, precision_recall_curve,
precision_score, recall_score, f1_score,
roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
@dataclass
class MODEL_TRAINER_CONFIG:
    model_path: str = os.path.join('artifacts','model.pkl')
class MODEL_TRAINER:
    def __init__(self):
        self.model_trainer_config = MODEL_TRAINER_CONFIG()
    def training_initalizer(self, train_array, test_array):
        try:
            logging.info('MODEL TRAINER STAGE STARTS >>>>')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            base_learner= [
                ('xgb', XGBClassifier(n_estimators=100, random_state=42)),
                ('lgbm', LGBMClassifier(n_estimators=100, random_state=42)),
                ('hgb', HistGradientBoostingClassifier(max_iter=100, random_state=42)),
                ('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('dtc', DecisionTreeClassifier(max_depth=10, random_state=42))
            ]
            final_learner = LogisticRegression()
            model = StackingClassifier(
            estimators= base_learner,
            final_estimator= final_learner,
            cv=5
            )
            model.fit(X_train, y_train)
            logging.info('Saving trained model')
            save_obj(self.model_trainer_config.model_path, model)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            # Evaluation
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)
            pr_curve = precision_recall_curve(y_test, y_proba)
            print(model)
            print(f'Accuracy: {accuracy}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}')
            print(f'ROC AUC: {roc_auc}')
            print(f'Confusion Matrix:\n{conf_matrix}')
            return accuracy
        except Exception as e:
            raise CustomException(e, sys)