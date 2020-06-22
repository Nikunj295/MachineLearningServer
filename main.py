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
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2)
    # df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    X=pd.DataFrame(X)
    y=pd.DataFrame(y)
    return X,y

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
    return jsonify(r2_score(y_test, y_pred))

@app.route('/logisticRegression')
def logistic():
    X,y = createData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    # X_test['Predicted'] = y_pred
    X_test['Original'] = y_test
    # return jsonify(r2_score(y_test, y_pred))
    
    params = request.args
    params_non_flat = params.to_dict(flat=False)
    temp = {k: v for k, v in params_non_flat.items()}
    print(type(temp))
    # end=None
    start = int(params["start"])
    if 'end' in temp:
        end = int(params["end"])
    else:
        end = None
    X_test = X_test[start:end][['Original','Predicted']]
    return X_test.to_json(orient='index')


if __name__ =='__main__':
    app.run(debug=True)