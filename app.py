from flask import Flask,render_template, request, redirect

app= Flask(__name__)

@app.route('/', methods=['GET','POST'])
def login():

@app.route('/User', methods=['GET','POST'])
def homepage():
    


app.run(host='0.0.0.0',port=5000)