from flask import Flask, render_template, url_for
from flask import request
from flask import jsonify, make_response

from sklearn.datasets import make_blobs
# from matplotlib import pyplot
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split    
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

def createData():
    X, y = make_blobs(n_samples=1000, centers=3, n_features=3)
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    return X,y


def normalize_query_param(value):
    return value if len(value) > 1 else value[0]


def get_params(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: normalize_query_param(v) for k, v in params_non_flat.items()}
    end = int(params["end"]) if 'end' in temp else None
    start = int(params["start"]) if 'start' in temp else None
    return start,end


@app.route('/linearRegression')
def linear():
    X,y = createData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    X_test['Predicted'] = y_pred
    X_test['Original'] = y_test

    # start,end = get_params(request.args)
    # X_test = X_test[start:end][['Original','Predicted']]
    return X_test.to_json(orient="index")


@app.route('/logisticRegression')
def logistic():
    X,y = createData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    X_test['Predicted'] = y_pred
    X_test['Original'] = y_test
    SCORE = jsonify(r2_score(y_test, y_pred))
    
    
    ##### HANDLE PARAMS #####
    start,end = get_params(request.args)
    X_test = X_test[start:end][['Original','Predicted']]
    
    ##### Output #####
    return X_test.to_json(orient='index')


if __name__ =='__main__':
    app.run(debug=True)