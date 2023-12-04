import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                  train_array[:, :-1],
                  train_array[:, -1],
                  test_array[:, :-1],
                  test_array[:, -1], 
            )
            

            models = {
            'Linear Regression':LinearRegression(),
            'Ridge Regression':Ridge(),
            'Lasso Regression':Lasso(),
            'Random Forest':RandomForestRegressor(),
            'k-Nearest Neighbors':KNeighborsRegressor(), 
            'Support Vector Machines':SVR(),
            'XGBoost':XGBRegressor(),
            'Gradient Boosting':GradientBoostingRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            'AdaBoost Regressor':AdaBoostRegressor()
            }
            
            model_report:dict=evaluate_models(X_train = X_train, y_train = y_train, 
                                              X_test = X_test, y_test = y_test, 
                                              models = models)
            

        ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e,sys)

 