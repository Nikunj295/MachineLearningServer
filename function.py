from sklearn.datasets import make_blobs
import pandas as pd

def createData(no_of_rows,no_of_columns,clust):
    X, y = make_blobs(n_samples=no_of_rows, centers=clust, n_features=no_of_columns,cluster_std=0.5,random_state=10)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    return X,y

def getMultipleValues(value):
    return value if len(value) > 1 else value[0]

def get_params(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    end = int(params["end"]) if 'end' in temp else None
    start = int(params["start"]) if 'start' in temp else None
    no_of_rows = int(params["rows"]) if 'rows' in temp else 100
    clust = int(params["clust"]) if 'clust' in temp else 2
    no_of_columns = int(params["cols"]) if 'cols' in temp else 2
    knear = int(params["knear"]) if 'knear' in temp else 5
    kernel = str(params['kernel']) if 'kernel' in temp else "linear"
    max_depth = int(params['max_depth']) if 'max_depth' in temp else None
    n_estimators = int(params['n_estimators']) if 'n_estimators' in temp else 50
    return [start, end, no_of_rows, no_of_columns, clust, knear, kernel,max_depth,n_estimators]

def get_algo(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    type_of_algo = params["algorithm"] if 'algorithm' in temp else None
    return type_of_algo
