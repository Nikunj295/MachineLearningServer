from flask import Flask, url_for, request, redirect
from Classification import classification
from Regression import regression
from function import createData
from flask_cors import CORS
from pymongo import MongoClient
import datetime
from sklearn import tree
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
import random
import base64
import datetime
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
from io import BytesIO

app = Flask(__name__)
CORS(app)
# app.register_blueprint(classification, url_prefix="/classification")
# app.register_blueprint(regression, url_prefix="/regression")

# app.config.from_pyfile('config.py')

client = MongoClient('mongodb+srv://nikunj:tetsu@dataframe.cbwqw.mongodb.net/User?retryWrites=true&w=majority')
client1 = MongoClient('mongodb+srv://nikunj:tetsu@dataframe.cbwqw.mongodb.net/Predefine?retryWrites=true&w=majority')


@app.route("/addId",methods=['GET','POST'])
def addId():
    db = client['User']
    ID = request.args.get("id")
    collection = db['Data']
    if collection.find({'_id':ID}).count() > 0: 
        pass
    else:
        mydict = { "_id": ID, 'data': {'X':"",'y':"","model":"","result":"", "pred":"","train":"",'test':"",'raw':''},'createdAt':datetime.datetime.utcnow()}
        collection.insert_one(mydict)
    return "Data Inserted"

@app.route("/create",methods=['GET','POST'])
def create():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    rows = int(dc.get('rows'))
    cols = int(dc.get('cols'))
    cluster = int(dc.get('cluster'))
    ty = dc.get('type')
    X = []
    y = []
    if ty == "classification":
        center_box = (0,500)
        X, y = datasets.make_blobs(n_samples=rows, centers=cluster, center_box=center_box, n_features=cols, cluster_std=10, random_state=random.randint(0,5000))
    if ty == "regression":
        X, y = datasets.make_regression(n_samples=rows, n_features=cols, noise=5)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    for i in range(X.shape[1],-1,-1):
        X = X.rename({ i : str(i+1) },axis=1)
    
    y = y.rename({0:'target'},axis=1)
    df1 = pd.concat([X.reset_index(drop='True'),y.reset_index(drop='True')],axis=1)
    desc = df1.describe().reset_index()
    col_name="index"
    first_col = desc.pop(col_name)
    desc.insert(0, col_name, first_col)
    db = client['User']
    collection = db['Data']
    raw = df1.to_dict('records')
    collection.update({'_id':userId},{ '$set': { 'data.raw': raw }})    
    return json.dumps( [json.loads(df1.to_json(orient="index")),json.loads(desc.to_json(orient="index")),] )


@app.route("/fetchData/<name>",methods=['GET','POST'])
def fetchData(name):
    db = client1['Predefine']
    collection = db[name]
    df1 = pd.DataFrame(list(collection.find({},{'_id':False,'index':False})))
    desc = df1.describe().reset_index()
    return json.dumps( [json.loads(df1.to_json(orient="index")),json.loads(desc.to_json(orient="index")),] )


@app.route('/create/selection',methods=['GET','POST'])
def createSelection():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    column = dc.get('item')
    db = client['User']
    collection = db['Data']
    x = list(collection.find({'_id':userId},{'_id':False}))
    df1 = pd.DataFrame(x[0]['data']['raw']) 

    X = df1[:][column]
    y = df1[:]['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    train = pd.concat([X_train.reset_index(drop='True'),y_train.reset_index(drop='True')],axis=1)
    test = pd.concat([X_test.reset_index(drop='True'),y_test.reset_index(drop='True')],axis=1)     
    train = train.to_dict('records')
    test = test.to_dict('records')
    
    collection.update({'_id':userId}, { '$set':{'data.train': train } })
    collection.update({'_id':userId}, { '$set':{'data.test': test } })
    column.append('target')
    df1 = df1[:][column]    
    return df1.to_json(orient="index")


@app.route("/selection",methods=['GET','POST'])
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
    collection.update({'_id':userId}, { '$set':{'data.train': train } })
    collection.update({'_id':userId}, { '$set':{'data.test': test } })
    column.append('target')
    df1 = df1[:][column]
    return df1.to_json(orient="index")

@app.route("/splitData",methods=['GET','POST'])
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

@app.route('/model',methods=['GET','POST'])
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
    
    return "Model Trained"

@app.route('/predict',methods=['GET','POST'])
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

@app.route('/visualize',methods=['GET','POST'])
def visualize():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get("id")
    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    result = pd.DataFrame(data[0]['data']['result']) 
    final = pd.DataFrame(data[0]['data']['pred']) 
    return json.dumps( [json.loads(final.to_json(orient="index")),json.loads(result.to_json(orient="index")),] )

@app.route("/getTree",methods=['GET','POST'])
def getTree():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    model = data[0]['data']['model']
    final = pd.DataFrame(data[0]['data']['pred']) 
    mdl = pickle.loads(model)
    fig = plt.figure(figsize=(20,20))
    tar = final[final.columns[-1]].unique().tolist()
    for i in range(len(tar)):
        tar[i] = str(tar[i])

    _ = tree.plot_tree(mdl, feature_names=final.columns,class_names=tar,filled=True)
    figfile = BytesIO()
    fig.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    figdata_png.decode("utf-8") 
    return figdata_png
    
@app.route("/boxplot",methods=['GET','POST'])
def boxplot():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    final = pd.DataFrame(data[0]['data']['result']) 

    li = []
    for i in final.columns:
        q2 = np.quantile(final[i], .50)
        q1 = np.quantile(final[i], .25)
        q3 = np.quantile(final[i], .75)
        mx = max(final[i])
        mn = min(final[i])
        li.append({"label":i,"y":[mn,q1,q3,mx,q2]})
    return json.dumps({"list":li})

@app.route("/corr",methods=['GET','POST'])
def corr():
    payload = request.args.get("payload")
    dc = json.loads(payload)
    userId = dc.get('id')
    db = client['User']
    collection = db['Data']
    data = list(collection.find({'_id':userId}))
    final = pd.DataFrame(data[0]['data']['result'])
    x = final.corr(method="pearson").reset_index().round(3) 
    return x.to_json(orient="index")

if __name__ =='__main__':
    app.run(debug=True)
