import pymysql

conn = pymysql.connect(
    host="localhost",
    user="root",
    password="password",
    database="university_db"
)

print("Connected successfully")

c = conn.cursor()

sql = "SELECT * FROM students;"
c.execute(sql)

result = c.fetchall()

for row in result:
    print(row)

conn.close()