from flask import Flask, url_for, request, redirect
from Classification import classification
from Regression import regression
from function import createData, get_params
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(classification, url_prefix="/classification")
app.register_blueprint(regression, url_prefix="/regression")


@app.route('/<learning>')
def selectAlgorithm(learning):
    if learning == "regression":
        return redirect(url_for('regression.home'))
    elif learning == "classification":
        return redirect(url_for('classification.home'))

if __name__ =='__main__':
    app.run(debug=True)

#,start=data_params[0],end=data_params[1], rows=data_params[2], cols=data_params[3], clust=data_params[4])