import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="Credentials"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM user_credentials")
myresult = mycursor.fetchall()

for x in myresult:
  print(x)
