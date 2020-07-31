from flask import Blueprint,request,redirect,url_for
from function import createData,get_algo, get_params, result, regressionData
from sklearn import linear_model, datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split    
import pandas as pd
from flask_cors import CORS
import json
regression = Blueprint('regression', __name__)
CORS(regression)

@regression.route("/")
def home():
    params = get_algo(request.args)
    if params is None:
        params = "linearRegression"
    data_params = get_params(request.args)
    if params == "linearRegression":
        return redirect(url_for('.linear',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]))
    elif params == "logisticRegression":
        return redirect(url_for('.logistic',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]))
    elif params == "ridge":
        return redirect(url_for('.ridge',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4],alpha=data_params[10]))
    else:
        return "select algo"

@regression.route('/linearRegression')
def linear():
    params = get_params(request.args)    
    X_train, X_test, y_train, y_test = regressionData(params[2],params[3])
    start, end = params[0], params[1]
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    return res.to_json(orient='index')

@regression.route("/logisticRegression")
def logistic():
    params = get_params(request.args)   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("log")
    print(params)
    return res.to_json(orient='index')

@regression.route("/ridge")
def ridge():
    params = get_params(request.args)
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = regressionData(params[2],params[3])
    ridgereg = linear_model.Ridge(alpha=params[10],normalize=True)
    ridgereg.fit(X_train,y_train)
    y_pred = ridgereg.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("ridge")
    print(params)
    return res.to_json(orient='index')

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df
