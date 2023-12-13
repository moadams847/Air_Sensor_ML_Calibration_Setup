
# ##--------------------------------------------------------------------------
import os
import sys
import pytz
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from datetime import datetime, timezone

from src.exception import CustomException

import random
random.seed(123)  

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            # test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # test_mae = mean_absolute_error(y_test, y_test_pred)

            # test_correlation = np.corrcoef(y_test, y_test_pred)[0, 1]

            report[list(models.keys())[i]] = test_model_score

            logging.info(report)

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# Function to convert datetime to milliseconds
def convert_to_milliseconds(FromDate, formatDate="%Y-%m-%d %H:%M:%S"):
    date_ = datetime.strptime(FromDate, formatDate).strftime(formatDate)
#     print(type(date_))
#     print(date_)
    
    date_format = datetime.strptime(date_, formatDate)
#     print(type(date_format))
#     print(date_format)

    
     #Set the timezone to UTC
    fromdate_utc = pytz.utc.localize(date_format)
#     print(type(fromdate_utc))
#     print(fromdate_utc)
    
    
    # Convert to milliseconds
    fromdate_utc_in_milliseconds = int(fromdate_utc.timestamp() * 1000)

    return fromdate_utc_in_milliseconds


# Function to convert milliseconds to datetime
def convert_to_datetime(milliseconds):
    timestamp_seconds = milliseconds / 1000
    datetime_obj = datetime.utcfromtimestamp(timestamp_seconds)
    return datetime_obj


def filter_data(df):
    try:
        # Combine all conditions into a single filtering operation
        df = df[
        (df['PM2_5'].between(1, 999)) &
        (df['PM2.5'].between(1, 999)) &
        (df['PM_10'].between(1, 999)) &
        (df['Temp'].between(22, 35)) &
        (df['RH'] > 0)
        ].reset_index(drop=True)
        return df
       
    except Exception as e:
        raise CustomException(e, sys)
    

def preprocess_data(df):
    try:
        # Calculate 'PM2_5-PM10'
        df['PM2_5-PM10'] = df['PM2_5'] - df['PM_10']

        # Extract month and hour from 'DataDate' column
        df['Month'] = df['DataDate'].dt.month
        df['Hour'] = df['DataDate'].dt.hour
        
        return df
    
    except Exception as e:
        raise CustomException(e, sys)

