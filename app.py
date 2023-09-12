from flask import Flask,render_template, request, redirect,url_for,session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import mysql.connector
import os
import re
import finance as fin

app= Flask(__name__)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Credentials",
)

app.config['SECRET_KEY'] = os.urandom(12).hex()

message=''
@app.route('/login', methods=['GET','POST'])
def login():
    message=''
    if request.method == 'POST':
        if 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            cursor = mydb.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user_credentials WHERE Username = %s AND Password = %s', (username, password, ))
            account = cursor.fetchone()
            if account:
                session['loggedin'] = True
                session['username'] = account[0]
                message = 'Logged in successfully !'
                return render_template('user.html', message = message)
                
            else:
                message = 'Invalid Credentials. Please try again.'
                return redirect(url_for('login'))
        
    return render_template('login.html', message=message)

@app.route('/register', methods =['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        if 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            cursor = mydb.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user_credentials WHERE Username = %s', (username,))
            account = cursor.fetchone()
            if account:
                message = 'Account already exists !'
            elif not re.match(r'[A-Za-z0-9]+', username):
                message = 'Username must contain only characters and numbers !'
            elif not username or not password:
                message = 'Please fill out the form !'
            else:
                cursor.execute('INSERT INTO user_credentials VALUES (%s, %s)', (username, password,))
                mydb.commit()
                message = 'You have successfully registered !'
        elif request.method == 'POST':
            message = 'Please fill out the form !'
    return render_template('register.html', message = message)

@app.route('/user', methods=['GET','POST'])
def homepage():
    if request.method == 'POST':
        values=list(request.form.values())
        if values[0]=='create':
            print('create selected')
            return redirect(url_for('creator'))
        elif values[0]=='view':
            return render_template('view.html')
    return render_template('user.html') 


@app.route('/create',methods=['GET','POST'])
def creator():
    if request.method=='POST':
        strategy=request.form['strategy']
        return redirect(url_for('port_strat',strat=strategy))
        #strategy=request.args.getlist('strategy')
    return render_template('create.html')
    
@app.route('/create/<strat>',methods=['GET','POST'])
def port_strat(strat):
    if request.method=='GET':
        print(strat)
    return strat
app.run(host='0.0.0.0',port=3306,debug=True)