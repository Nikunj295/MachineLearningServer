from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import pandas as pd
import random


def createData(no_of_rows, no_of_columns, clust):
    center_box = (10,500)
    X, y = make_blobs(n_samples=no_of_rows, centers=clust,center_box=center_box, n_features=no_of_columns,cluster_std=100,random_state=random.randint(0,5000))
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    return X_train, X_test, y_train, y_test

def getMultipleValues(value):
    return value if len(value) > 1 else value[0]

def get_params(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    end = int(params["end"]) if 'end' in temp and params['end']!=''else None
    start = int(params["start"]) if 'start' in temp and params['start']!='' else None
    no_of_rows = int(params["rows"]) if 'rows' in temp and params['rows']!='' else 100
    clust = int(params["clust"]) if 'clust' in temp and params['clust']!=''else 2
    no_of_columns = int(params["cols"]) if 'cols' in temp and params['cols']!='' else 2
    knear = int(params["knear"]) if 'knear' in temp and params['knear']!='' else 5

    if 'max_depth' in temp and params['max_depth']!='':
        max_depth = int(params['max_depth'])
    else: 
        max_depth = None

    if 'kernel' in temp and params['kernel']!='':
        kernel = str(params['kernel'])
    else: 
        kernel = "linear"

    n_estimators = int(params['n_estimators']) if 'n_estimators' in temp else 50
    alpha = int(params['alpha']) if 'alpha' in temp else 1
    degree = int(params['degree']) if 'degree' in temp else 50
    return [start, end, no_of_rows, no_of_columns, clust, knear, kernel,max_depth,n_estimators,degree,alpha]

def get_algo(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    type_of_algo = params["algorithm"] if 'algorithm' in temp else None
    return type_of_algo

def result(X_test,y_test,y_pred):
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    return df2

def regressionData(no_of_rows,no_of_columns):
    X, y = make_regression(n_samples=no_of_rows, n_features=no_of_columns, noise=5)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=15)
    return X_train, X_test, y_train, y_test

    
