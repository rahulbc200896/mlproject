import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer

@dataclass
class data_ingestion_config:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

class data_ingestion:
    def __init__(self):
        self.data_ingestion = data_ingestion_config()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.data_ingestion.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,header=True,index=False)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion.train_data_path,header=True,index=False)
            test_set.to_csv(self.data_ingestion.test_data_path,header=True,index=False)

            logging.info("Data ingestion Completed")

            return(self.data_ingestion.train_data_path,self.data_ingestion.test_data_path)
        
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
if __name__ == '__main__':
    obj = data_ingestion()
    train_data,test_data = obj.initiate_data_ingestion()
    data_transformer = Data_Transformation()
    train_arr,test_arr,_ = data_transformer.initiate_data_transformation(train_data,test_data)
    model_trainer = Model_Trainer()
    best_model_name,best_model_score = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print("Best model: ",best_model_name)
    print("Model Accuracy: ",best_model_score)