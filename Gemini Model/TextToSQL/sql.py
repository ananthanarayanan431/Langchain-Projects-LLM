
#loading environment variable
from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
import os 
import sqlite3

import google.generativeai as genai 

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# print(genai)
#function and provides queries as response
def get_gemini_response(question,prompt):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content([prompt[0],question])
    return response.text

#function to retrieve data from database
def read_sql_query(sql,db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

#Define your prompt
prompt=[
    """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name STUDENT and has the following columns - NAME, CLASS, 
    SECTION \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM STUDENTS ;
    \nExample 2 - Tell me all the students studying in Data Science class?, 
    the SQL command will be something like this SELECT * FROM STUDENTS 
    where CLASS="Data Science"; 
    also the sql code should not have ``` in beginning or end and sql word in output

    """


]


#streamlit app

st.set_page_config(page_title ="I can Retrieve any SQL query")
st.header("Gemini App to retrieve SQL Data")

question = st.text_input("Input: ",key="input")
submit = st.button("Ask the Question")

if submit:
    response = get_gemini_response(question,prompt)
    print(response)
    response=read_sql_query(response,"students.db")
    st.subheader("The Response is")
    for row in response:
        print(row[0])
        st.subheader(row[0])