from flask import Flask,render_template, request, redirect,url_for

app= Flask(__name__)

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)

#@app.route('/User', methods=['GET','POST'])
#def homepage():



app.run(host='0.0.0.0',port=5000)