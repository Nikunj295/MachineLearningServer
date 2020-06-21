from flask import Flask, render_template, url_for
from flask import request
from flask import jsonify, make_response

app = Flask(__name__)

@app.route('/')
def index():
    response_body = {
        "message": "JSON received!"
    }
    res = make_response(jsonify(response_body), 200)
    return res

if __name__ =='__main__':
    app.run(debug=True)