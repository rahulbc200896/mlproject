import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model  import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRFRegressor
from src.utils import save_obj,evaluate_model

class model_trainer_config:
    model_trainer_file_path = os.path.join('artifacts','model.pkl')

class Model_Trainer:
    def __init__(self):
        self.model_trainer = model_trainer_config()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Loaading the train and test data into x_train,x_test,y_train,y_test")
            x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models = {
                "linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Elasticnet": ElasticNet(),
                "Knearest Regressor": KNeighborsRegressor(),
                "SVR": SVR(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Gradient Bosst": GradientBoostingRegressor(),
                "Xgboost": XGBRFRegressor()
            }
            
            logging.info("Feeding train and test data to evaluate model")
            Model_report:dict = evaluate_model(x_train,x_test,y_train,y_test,models)
            best_model_score = max(sorted(Model_report.values()))
            best_model_name = list(Model_report.keys())[list(Model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info("Fetching the best model and r2 score")
            if best_model_score<0.6:
                raise CustomException("Best model not found")
            else:
                save_obj(self.model_trainer.model_trainer_file_path,best_model)
            
            return (best_model_name,best_model_score)
        except Exception as e:
            raise CustomException(e,sys)