import sqlite3

connection = sqlite3.connect("students.db")

#cursor creation
cursor = connection.cursor()

table_info = """
Create table if not exists STUDENTs(NAME VARCHAR(25), CLASS VARCHAR(25), SECTION VARCHAR(25), MARKS INT);
"""

cursor.execute(table_info)

#insert records

cursor.execute('''Insert into STUDENTS values('Anantha','Artificial Intelligence','A',90)''')
cursor.execute('''Insert into STUDENTS values('Narayanan','Artificial Intelligence','A',100)''')
cursor.execute('''Insert into STUDENTS values('Ajay','Devops','B',87)''')
cursor.execute('''Insert into STUDENTS values('Harsha Vardhan','Communication','B',76)''')
cursor.execute('''Insert into STUDENTS values('pratheep','Computer Science','A',83)''')

#display

print("The Inserted records are")
data = cursor.execute('''select * from STUDENTS''');
for row in data:
    print(row)

connection.commit()
connection.close()