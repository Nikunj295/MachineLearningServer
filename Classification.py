from flask import Blueprint,request,redirect,url_for
from function import createData, get_params, get_algo
from sklearn import linear_model
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
    
import pandas as pd
import numpy as np

classification = Blueprint('classification', __name__)

@classification.route("/")
def home():
    params = get_algo(request.args)
    if params is None:
        params = "logisticRegression"
    data_params = get_params(request.args)
    if params == "logisticRegression":
        return redirect(url_for('.logistic',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]))
    elif params == "knear":
        return redirect(url_for('.knear',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], knear=data_params[5]))
    elif params == "svm":
        return redirect(url_for('.svm',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], knear=data_params[5]))
    elif params == "naive":
        return redirect(url_for('.naive',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], knear=data_params[5]))
    elif params == "dtree":
        return redirect(url_for('.dtree',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], knear=data_params[5]))
    elif params == "rtree":
        return redirect(url_for('.rtree',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], knear=data_params[5]))
    else:
        return "select algo"

@classification.route("/logisticRegression")
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

@classification.route("/knear")
def knear():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    classifier = KNeighborsClassifier(n_neighbors=params[5])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("KNEAR")
    return df2.to_json(orient='index')

@classification.route("/svm")
def svm():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = SVC(kernel='linear') 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("SVM")
    return df2.to_json(orient='index')

@classification.route("/naive")
def naive():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("NAIVE")
    return df2.to_json(orient='index')

@classification.route("/dtree")
def dtree():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("DTREE")
    print(accuracy_score(y_test,y_pred))
    return df2.to_json(orient='index')

@classification.route("/rtree")
def rtree():
    params = get_params(request.args)
    X,y = createData(params[2],params[3],params[4])   
    start, end = params[0], params[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,np.ravel(y_train))
    y_pred=clf.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_test = y_test.rename(columns={0:"Original"})
    y_pred = y_pred.rename(columns={0:"Predicted"})
    df1 = pd.concat([y_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    df2 = pd.concat([X_test.reset_index(drop='True'),df1.reset_index(drop='True')],axis=1)
    ##### HANDLE PARAMS #####
    df2 = df2[start:end]
    ##### Output #####
    print("RTREE")
    print(accuracy_score(y_test,y_pred))
    return df2.to_json(orient='index')

