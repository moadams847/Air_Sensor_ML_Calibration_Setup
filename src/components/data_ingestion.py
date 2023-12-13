import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from src.utils import preprocess_data, filter_data

import random
random.seed(123)  

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            logging.info('Read the dataset as dataframe')
            df = pd.read_csv('notebook/data/merge_raw_data_Teledyne/calibration_data_sets/merged_Teledyne_data_weather_data_ENE00933.csv', parse_dates=['DataDate'])
            df.rename(columns={'PM10': 'PM_10'}, inplace=True)
            df = df[['DataDate','PM2.5','PM2_5', 'PM_10', 'RH', 'Temp']]

            logging.info('Filter rows')
            df = filter_data(df)

            logging.info('Engineer columns')
            df = preprocess_data(df)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    best_model_name, r_squred, correlation_, rmse_, mae_ = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(f'model_name:{best_model_name}')
    print(f'R-squared:{r_squred}')
    print(f'correlation:{correlation_}')
    print(f'RMSE:{rmse_}')
    print(f'MAE:{mae_}')


   