from flask import Blueprint,request,redirect,url_for
from function import createData,get_algo, get_params
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split    
import pandas as pd

regression = Blueprint('regression', __name__)

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
    # elif params == "poly":
    #     return redirect(url_for('.poly',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4],degree=data_params[9]))
    elif params == "ridge":
        return redirect(url_for('.ridge',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4],alpha=data_params[10]))

@regression.route('/linearRegression')
def linear():
    params = get_params(request.args)    
    X,y = createData(params[2],params[3],params[4])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    start, end = params[0], params[1]
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("linear")
    return df2.to_json(orient='index')

@regression.route("/logisticRegression")
def logistic():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)   
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predicted = pd.DataFrame(y_pred,columns=['Predicted'])
    y_test = y_test.rename(columns={0:"Original"})
    df1 = pd.concat([y_test.reset_index(drop='True'),predicted.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("LOGISTIC")
    return df2.to_json(orient='index')

@regression.route("/poly")
def poly():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)   
    poly = PolynomialFeatures(degree = params[9]) 
    X_poly = poly.fit_transform(X_train) 
    poly.fit(X_poly, y_train) 
    lin2 = linear_model.LinearRegression() 
    lin2.fit(X_poly, y_train)
    y_pred = lin2.predict(poly.fit_transform(X_test)) 
    predicted = pd.DataFrame(y_pred,columns=['Predicted'])
    y_test = y_test.rename(columns={0:"Original"})
    df1 = pd.concat([y_test.reset_index(drop='True'),predicted.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("POLY")
    print(request.args)
    return df2.to_json(orient='index')

@regression.route("/ridge")
def ridge():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)   
    ridgereg = Ridge(alpha=params[10],normalize=True)
    ridgereg.fit(X_train,y_train)
    y_pred = ridgereg.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("ridge")
    return df2.to_json(orient='index')