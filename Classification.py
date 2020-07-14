from flask import Blueprint,request,redirect,url_for, jsonify
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
from sklearn import datasets

import json
import pandas as pd
import numpy as np

classification = Blueprint('classification', __name__)
CORS(classification)

@classification.route("/")
def home():
    params = get_algo(request.args)
    if params is None or "":
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
    
@classification.route("/logisticRegression")
def logistic():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    model = linear_model.LogisticRegression(random_state=0)
    model.fit(X_train, np.ravel(y_train))
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("log")
    print(params)
    return res.to_json(orient='index')

@classification.route("/knear")
def knear():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    classifier = KNeighborsClassifier(n_neighbors=params[5])
    classifier.fit(X_train, np.ravel(y_train))
    y_pred = classifier.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("knear")
    
    print(params)
    return res.to_json(orient='index')

@classification.route("/svm")
def svm():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    clf = SVC(kernel=params[6]) 
    clf.fit(X_train, np.ravel(y_train))
    y_pred = clf.predict(X_test) 
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("svm")
    print(params)
    return res.to_json(orient='index')

@classification.route("/naive")
def naive():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    model = GaussianNB()
    model.fit(X_train,np.ravel(y_train))
    y_pred = model.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("naive")
    print(params)
    return res.to_json(orient='index')

@classification.route("/dtree")
def dtree():
    params = get_params(request.args)
    X_train, X_test, y_train, y_test = createData(params[2],params[3],params[4])
    start, end = params[0], params[1]
    clf = DecisionTreeClassifier(max_depth= params[7])
    clf = clf.fit(X_train,np.ravel(y_train))
    y_pred = clf.predict(X_test)
    res = result(X_test, y_test, y_pred)
    res = res[start:end]
    print("dtree")
    print(params)
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
    print(params)
    return res.to_json(orient='index')

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

@classification.route("/fetchData/<name>",methods=['GET','POST'])
def fetchData(name):
    if name == "iris":
        iris_data = datasets.load_iris()
        df = sklearn_to_df(iris_data)
        df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
        desc = df.describe()
        desc = desc.reset_index()
        return json.dumps( [json.loads(df.to_json(orient="index")),json.loads(desc.to_json(orient="index")) ] )

    elif name == "boston":
        boston_data = datasets.load_boston()
        df = sklearn_to_df(boston_data)
        desc = df.describe()
        desc = desc.reset_index()
        return json.dumps( [json.loads(df.to_json(orient="index")),json.loads(desc.to_json(orient="index")) ] )

    elif name == "digits":
        digits_data = datasets.load_digits()
        df = sklearn_to_df(digits_data)
        desc = df.describe()
        desc = desc.reset_index()
        return json.dumps( [json.loads(df.to_json(orient="index")),json.loads(desc.to_json(orient="index")) ] )

    elif name == "breast":
        breast_cancer_data = datasets.load_breast_cancer()
        df = sklearn_to_df(breast_cancer_data)
        desc = df.describe()
        desc = desc.reset_index()
        return json.dumps( [json.loads(df.to_json(orient="index")),json.loads(desc.to_json(orient="index")) ] )

    elif name == "wine":
        wine_data = datasets.load_wine()
        df = sklearn_to_df(wine_data)
        desc = df.describe()
        desc = desc.reset_index()
        return json.dumps( [json.loads(df.to_json(orient="index")),json.loads(desc.to_json(orient="index")) ] )

    