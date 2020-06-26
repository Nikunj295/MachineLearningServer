from flask import Blueprint,request,redirect,url_for
from function import createData, get_params, get_algo,result
from sklearn import linear_model
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS
    
import pandas as pd
import numpy as np

classification = Blueprint('classification', __name__)
CORS(classification)

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
        return redirect(url_for('.svm',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], kernel=data_params[6]))
    elif params == "naive":
        return redirect(url_for('.naive',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4]))
    elif params == "dtree":
        return redirect(url_for('.dtree',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], max_depth=data_params[7]))
    elif params == "rtree":
        return redirect(url_for('.rtree',start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4], n_estimators=data_params[8]))
    else:
        return "select algo"

@classification.route("/logisticRegression")
def logistic():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("log")
    return res.to_json(orient='index')

@classification.route("/knear")
def knear():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    classifier = KNeighborsClassifier(n_neighbors=params[5])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("knear")
    return res.to_json(orient='index')

@classification.route("/svm")
def svm():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    clf = SVC(kernel=params[6]) 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test) 
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("svm")
    return res.to_json(orient='index')

@classification.route("/naive")
def naive():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    model = GaussianNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("naive")
    return res.to_json(orient='index')

@classification.route("/dtree")
def dtree():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    clf = DecisionTreeClassifier(max_depth= params[7])
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("dtree")
    return res.to_json(orient='index')

@classification.route("/rtree")
def rtree():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    clf=RandomForestClassifier(n_estimators=params[8])
    clf.fit(X_train,np.ravel(y_train))
    y_pred=clf.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("rtree")
    return res.to_json(orient='index')
