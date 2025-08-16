import os
import sys
import joblib
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj
from dataclasses import dataclass
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
roc_auc_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

@dataclass
class MODEL_TRAINER_CONFIG:
    model_path: str = os.path.join('artifacts','model.pkl')

class MODEL_TRAINER:
    def __init__(self, resampling_strategy='smote'):
        self.model_trainer_config = MODEL_TRAINER_CONFIG()
        self.resampling_strategy = resampling_strategy 

    def training_initalizer(self, train_array, test_array):
        try:
            logging.info('MODEL TRAINER STAGE STARTS >>>>')
            logging.info('Train Test Split')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Core learners
            xgb = XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'
            )
            hgb = HistGradientBoostingClassifier(class_weight='balanced')
            rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            knn = KNeighborsClassifier(n_neighbors=7)
            ada = AdaBoostClassifier(n_estimators=100, random_state=42)

            if self.resampling_strategy == 'smote':
                resampler = SMOTE(random_state=42)
            else:
                resampler = None

            estimators = [
                ('xgb', xgb),
                ('hgb', hgb),
                ('rf', rf),
                ('knn', knn),
                ('ada', ada)
            ]
            
            final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

            if resampler:
                model = ImbPipeline([
                    ('resample', resampler),
                    ('stack', StackingClassifier(
                        estimators=estimators,
                        final_estimator=final_estimator,
                        cv=5,
                        stack_method='predict_proba'
                    ))
                ])
            else:
                model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=5,
                    stack_method='predict_proba'
                )

            model.fit(X_train, y_train)

            logging.info('Saving trained model')
            save_obj(self.model_trainer_config.model_path, model)
            
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.30).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            conf_matrix = confusion_matrix(y_test, y_pred)

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
