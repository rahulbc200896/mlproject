import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging

def save_obj(file_path,obj):
    try:
        logging.info("Preprocessor pkl generation started")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj,file)
        logging.info("Preprocessor pkl file created")
    except Exception as e:
        raise CustomException(e,sys)
