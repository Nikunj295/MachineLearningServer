from flask import Flask, render_template, url_for
from flask import request,redirect
from flask import jsonify, make_response
from sklearn.datasets import make_blobs
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

def getMultipleValues(value):
    return value if len(value) > 1 else value[0]

def get_params(params):
    params_non_flat = params.to_dict(flat=False)
    temp = {k: getMultipleValues(v) for k, v in params_non_flat.items()}
    end = int(params["end"]) if 'end' in temp else None
    start = int(params["start"]) if 'start' in temp else None
    type_of_algo = params["algorithm"] if 'algorithm' in temp else "linearRegression"
    return [start, end,type_of_algo]

@app.route('/')
def selectAlgorithm():
    params = get_params(request.args)    
    if params[2]=="linearRegression":
        return redirect(url_for('linear',start=params[0],end=params[1]))

    elif params[2]=="logisticRegression":
        return redirect(url_for('logistic',start=params[0],end=params[1]))

@app.route('/linearRegression')
def linear():
    X,y = createData()
    params = get_params(request.args)    
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
    print(r2_score(y_test,predicted))
    return df2.to_json(orient="index")


@app.route('/logisticRegression')
def logistic():
    X,y = createData()
    params = get_params(request.args)   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    predicted = pd.DataFrame(y_pred,columns=['predicted'])
    
    # temp = pd.DataFrame(y_test,columns=['og'])
    y_test = y_test.rename(columns={0:"Original"})

    df1 = pd.concat([y_test.reset_index(drop='True'),predicted.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    
    start, end = params[0], params[1]
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    
    ##### Output #####
    print("logistic")
    print(r2_score(y_test,predicted))
    return df2.to_json(orient='index')


if __name__ =='__main__':
    app.run(debug=True)