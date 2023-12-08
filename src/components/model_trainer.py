import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, SGDRegressor
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

            'sgd':SGDRegressor(),
            # 'Ridge Regression':Ridge(),
            # 'Lasso Regression':Lasso(),
            'Random Forest':RandomForestRegressor(),
            # 'k-Nearest Neighbors':KNeighborsRegressor(), 
            # 'Support Vector Machines':SVR(),
            'XGBoost':XGBRegressor(),
            # 'Gradient Boosting':GradientBoostingRegressor(),
            # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            # 'AdaBoost Regressor':AdaBoostRegressor()
            }


            params={

                 # sgd
                "sgd":{
                    # 'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                    'penalty': ['l2', 'l1', 'elasticnet'],
                    'alpha': [0.0001, 0.001, 0.01, 0.1],
                    # 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive']
                },

                # # Ridge Regression
                # 'Ridge Regression': {
                #     'alpha': [0.001, 0.01, 0.1, 1, 10, 100] 
                # },                
            
                # # Lasso Regression
                # 'Lasso Regression': {
                #     'alpha': [0.001, 0.01, 0.1, 1, 10, 100] 
                # },

                # # k-Nearest Neighbors
                #  'k-Nearest Neighbors': {
                #  'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
                #  'weights': ['uniform', 'distance'],  # Weight function used in prediction
                #  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                # },

                # #Support Vector Machines
                # 'Support Vector Machines': {
                # 'C': [0.1, 1, 10],  # Regularization parameter
                # 'kernel': ['linear', 'rbf'],  # Type of kernel
                # 'gamma': ['scale', 'auto', 0.1, 1]
                # },

                # Random Forest
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                # # Gradient Boosting
                # "Gradient Boosting":{
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate':[.1,.01,.05,.001],
                #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [8,16,32,64,128,256]
                # },

                # XGBoost
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                # # CatBoosting Regressor
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },

                # # AdaBoost Regressor
                # "AdaBoost Regressor":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
                
            }

            model_report:dict=evaluate_models(X_train = X_train, y_train = y_train, 
                                              X_test = X_test, y_test = y_test, 
                                              models = models, param=params)
            

        ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            logging.info(best_model_score)

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[ 
            list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            test_r2_square = r2_score(y_test, predicted)

            test_correlation = np.corrcoef(y_test, predicted)[0, 1]

            test_rmse = np.sqrt(mean_squared_error(y_test, predicted))

            test_mae = mean_absolute_error(y_test, predicted)

            return best_model_name, test_r2_square, test_correlation, test_rmse, test_mae
        
        except Exception as e:
            raise CustomException(e,sys)

 