import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import sys
from dataclasses import dataclass
from src.utils import save_obj

@dataclass
class data_transformation_config:
    preprocessor_file_obj_path = os.path.join('artifacts','preprocessor.pkl')

class Data_Transformation:
    def __init__(self):
        self.data_transformation = data_transformation_config()
    
    def get_data_transformation_object(self):
        try:
            num_features = ['reading_score','writing_score']
            cat_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one hot encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical Features: {}".format(cat_features))
            logging.info("Numerical Features: {}".format(num_features))

            preprocessor = ColumnTransformer([
                ('numerical features',num_pipeline,num_features),
                ('categorical features',cat_pipeline,cat_features)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_data,test_data):
        try:
            
            self.data_transformer = self.get_data_transformation_object()
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)
            logging.info("Reading the train and test data completed")
            logging.info("Obtaining Preprocessing object")
            target_feature = 'math_score'
            train_input_data = train_data.drop(columns=[target_feature],axis=1)
            train_target_data = train_data[target_feature]

            test_input_data = test_data.drop(columns=[target_feature],axis=1)
            test_target_data = test_data[target_feature]

            train_input_data_arr = self.data_transformer.fit_transform(train_input_data)
            test_input_data_arr = self.data_transformer.transform(test_input_data)

            train_arr = np.c_[train_input_data_arr,np.array(train_target_data)]
            test_arr = np.c_[test_input_data_arr,np.array(test_target_data)]
            save_obj(
                self.data_transformation.preprocessor_file_obj_path,
                self.data_transformer
            )
            return(train_arr,
                   test_arr,
                   self.data_transformation.preprocessor_file_obj_path)
        except Exception as e:
            raise CustomException(e,sys)
