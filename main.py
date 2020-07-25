from flask import Flask, url_for, request, redirect
from Classification import classification
from Regression import regression
from function import createData, get_params
from flask_cors import CORS
from pymongo import MongoClient
import datetime

app = Flask(__name__)
CORS(app)
app.register_blueprint(classification, url_prefix="/classification")
app.register_blueprint(regression, url_prefix="/regression")
client = MongoClient('mongodb+srv://nikunj:tetsu@dataframe.cbwqw.mongodb.net/User?retryWrites=true&w=majority')
db = client['User']

@app.route("/addId",methods=['GET','POST'])
def addId():
    ID = request.args.get("id")
    collection = db['Data']
    algo = "iris"
    mydict = { "_id": ID, 'data': {'X':"",'y':"","model":""},'createdAt':datetime.datetime.utcnow()}
    collection.insert_one(mydict)
    return "Data Inserted"

@app.route('/<learning>')
def selectAlgorithm(learning):
    if learning == "regression":
        return redirect(url_for('regression.home'))
    elif learning == "classification":
        return redirect(url_for('classification.home'))

if __name__ =='__main__':
    app.run(debug=True)
