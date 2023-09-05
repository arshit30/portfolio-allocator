from flask import Flask,render_template, request, redirect,url_for
from flask_mysqldb import MySQL
import MySQLdb.cursors

app.secret_key = 'your secret key'
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your password'
app.config['MYSQL_DB'] = 'geeklogin'

mysql = MySQL(app)

app= Flask(__name__)

@app.route('/')
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = % s AND password = % s', (username, password, ))
        account = cursor.fetchone()
        if account:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            message = 'Logged in successfully !'
            return render_template('index.html', message = message)
            
        else:
            message = 'Invalid Credentials. Please try again.'
            return redirect(url_for('home'))
        
    return render_template('login.html', message=message)

@app.route('/register', methods =['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        if 'username' in request.form and 'password' in request.form:
            username = request.form['username']
            password = request.form['password']
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM accounts WHERE username = % s', (username, ))
            account = cursor.fetchone()
            if account:
                message = 'Account already exists !'
            elif not re.match(r'[A-Za-z0-9]+', username):
                message = 'Username must contain only characters and numbers !'
            elif not username or not password:
                message = 'Please fill out the form !'
            else:
                cursor.execute('INSERT INTO accounts VALUES (NULL, % s, % s)', (username, password, ))
                mysql.connection.commit()
                message = 'You have successfully registered !'
        elif request.method == 'POST':
            message = 'Please fill out the form !'
    return render_template('register.html', message = message)

#@app.route('/User', methods=['GET','POST'])
#def homepage():



app.run(host='0.0.0.0',port=5000)