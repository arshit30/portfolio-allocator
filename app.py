from flask import Flask,render_template, request, redirect,url_for,session,jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import mysql.connector
import os
import re
import finance as fin
import portfolio as pf
from healthcheck import HealthCheck
from test import *

app= Flask(__name__)
health = HealthCheck()

mydb = mysql.connector.connect(
  host="uyu7j8yohcwo35j3.cbetxkdyhwsb.us-east-1.rds.amazonaws.com",
  user="ln90fus9zps3kf5c",
  password="nm089pvb9w9821bx",
  database="mr76786mt2aisvgx",
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
            cursor.execute('SELECT * FROM users WHERE Username = %s AND Password = %s', (username, password, ))
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
            cursor.execute('SELECT * FROM users WHERE Username = %s', (username,))
            account = cursor.fetchone()
            if account:
                message = 'Account already exists !'
            elif not re.match(r'[A-Za-z0-9]+', username):
                message = 'Username must contain only characters and numbers !'
            elif not username or not password:
                message = 'Please fill out the form !'
            else:
                cursor.execute('INSERT INTO users VALUES (%s, %s)', (username, password,))
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
    return render_template('create.html')
    
@app.route('/create/<strat>',methods=['GET','POST'])
def port_strat(strat):
    if request.method=='GET':
        pf.data_collection()

        strategies_inscope(strat)  #tests if the strategy searched for  is supported

        pf_weights=pf.create_pf(strategy=strat)

        weight_constraint(pf_weights) # test all weights sum up to 1

        portfolio=pf.pf_results(weights=pf_weights)
    return jsonify(portfolio)

app.add_url_rule("/health", "healthcheck", view_func=lambda: health.run())

if __name__ == "__main__":
        app.run(host='0.0.0.0')