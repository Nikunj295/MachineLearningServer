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
from sklearn.feature_selection import RFE
from pymongo import MongoClient

import datetime
import pickle
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
    db = client1['Predefine']
    collection = db[name]
    df1 = pd.DataFrame(list(collection.find({},{'_id':False,'index':False})))
    desc = df1.describe().reset_index()
    return json.dumps( [json.loads(df1.to_json(orient="index")),json.loads(desc.to_json(orient="index")),] )

@classification.route("/selection",methods=['GET','POST'])
def selection():
    db = client1['Predefine']
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    column = dc.get('item')
    dataSet = dc.get('dataset')
    collection = db[dataSet]

    df1 = pd.DataFrame(list(collection.find({},{'_id':False,'index':False}))) 
    X = df1[:][column]
    y = df1[:]['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train = pd.concat([X_train.reset_index(drop='True'),y_train.reset_index(drop='True')],axis=1)
    test = pd.concat([X_test.reset_index(drop='True'),y_test.reset_index(drop='True')],axis=1)     
    train = train.to_dict('records')
    test = test.to_dict('records')

    db = client['User']
    collection = db['Data']
    collection.update({'_id':userId}, { "_id": userId, 'data': { 'train' : train , 'test' : test , 'model' : "" } ,'createdAt': datetime.datetime.utcnow()})
    column.append('target')
    df1 = df1[:][column]    
    return df1.to_json(orient="index")

@classification.route("/splitData",methods=['GET','POST'])
def splitData():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    
    db = client['User']
    collection = db['Data']
    temp = collection.find({'_id':userId})
    array = list(temp)
    train = pd.DataFrame(array[0]['data']['train']) 
    test = pd.DataFrame(array[0]['data']['test']) 
    return json.dumps( [json.loads(train.to_json(orient="index")),json.loads(test.to_json(orient="index")),] )

@classification.route('/model',methods=['GET','POST'])
def model():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    algorithm = dc.get("algorithm")
    userId = dc.get("id")

    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    train = pd.DataFrame(data[0]['data']['train'])
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]]

    if algorithm == "logisticRegression":
        model = linear_model.LogisticRegression()
        model.fit(X_train, np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print('log')
    
    elif algorithm == "knear":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print("knear")

    elif algorithm == "naive":
        model = GaussianNB()
        model.fit(X_train,np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print("naive")
    
    elif algorithm == "dtree":
        model = DecisionTreeClassifier()
        model.fit(X_train,np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print("dtree")
    
    elif algorithm == "rtree":
        model=RandomForestClassifier(n_estimators=50)
        model.fit(X_train,np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print('rtree')

    #both
    elif algorithm == "svm":
        model = SVC(kernel='linear') 
        model.fit(X_train, np.ravel(y_train))
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print("svm")

    # Regression Algorithm
    elif algorithm == 'linearRegression':
        model = linear_model.LinearRegression()
        model.fit(X_train, y_train)
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print('linear')

    elif algorithm == 'logisticRegression':
        model = linear_model.LogisticRegression(random_state=0)
        model.fit(X_train, y_train)
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print('logR')
    
    elif algorithm == 'ridge':
        model = linear_model.Ridge(normalize=True)
        model.fit(X_train,y_train)
        pickled_model = pickle.dumps(model)
        collection.update(  { '_id':userId} , { '$set': { 'data.model' : pickled_model  } } )
        print('ridge')
    

    return "From model"

@classification.route('/predict',methods=['GET','POST'])
def predicted():
    
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get("id")
    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    model = data[0]['data']['model']
    test = pd.DataFrame(data[0]['data']['test']) 

    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]]
    mdl = pickle.loads(model)
    y_pred = mdl.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.rename(columns = {0:'Predicted'}, inplace = True) 
    
    just = pd.concat([X_test.reset_index(drop='True'),y_pred.reset_index(drop='True')],axis=1)
    result = pd.concat([y_pred.reset_index(drop='True'),y_test.reset_index(drop='True')],axis=1)
    final = pd.concat([X_test.reset_index(drop='True'),result.reset_index(drop='True')],axis=1)
    j = just.to_dict('records')
    f = final.to_dict('records')
    collection.update(  { '_id':userId} , { '$set': { 'data.result' : f  } } )
    collection.update(  { '_id':userId} , { '$set': { 'data.pred' : j  } } )

    return json.dumps( [json.loads(just.to_json(orient="index")),json.loads(final.to_json(orient="index")),] )
