from flask import Flask, render_template, url_for
from flask import request,redirect
from flask import jsonify, make_response
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

def createData(no_of_rows,no_of_columns,clust):
    X, y = make_blobs(n_samples=no_of_rows, centers=clust, n_features=no_of_columns)
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
    return [start, end, no_of_rows, no_of_columns, clust]

def get_algo(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    type_of_algo = params["algorithm"] if 'algorithm' in temp else "linearRegression"
    return type_of_algo


################################## ROUTES ######################################

@app.route('/')
def selectAlgorithm():
    params = get_algo(request.args) 
    data_params = get_params(request.args)
    if params == "linearRegression":
        return redirect(url_for('linear',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]  ))

    elif params == "logisticRegression":
        return redirect(url_for('logistic',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]))

@app.route('/linearRegression')
def linear():
    params = get_params(request.args)    
    X,y = createData(params[2],params[3],params[4])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predicted = pd.DataFrame(y_pred,columns=['predicted'])
    y_test = y_test.rename(columns={0:"Original"})
    df1 = pd.concat([y_test.reset_index(drop='True'),predicted.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[params[0]:params[1]]
    ##### Output #####
    print("linear")
    return df2.to_json(orient="index")


@app.route('/logisticRegression')
def logistic():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predicted = pd.DataFrame(y_pred,columns=['predicted'])
    y_test = y_test.rename(columns={0:"Original"})
    df1 = pd.concat([y_test.reset_index(drop='True'),predicted.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("logistic")
    return df2.to_json(orient='index')


if __name__ =='__main__':
    app.run(debug=True)