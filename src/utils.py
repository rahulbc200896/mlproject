import os
import sys
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model  import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        logging.info("pkl generation started")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj,file)
        logging.info("pkl file created")
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,x_test,y_train,y_test,models,params):
    try:
        model_report = {}
        logging.info("Model training started")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            gs = GridSearchCV(estimator=model,param_grid=param,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_test_pred = model.predict(x_test)
            r2score = r2_score(y_test,y_test_pred)

            model_report[list(models.keys())[i]] = r2score
        
        logging.info("Evaluating the accuracy of models completed")
        return model_report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_data(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)