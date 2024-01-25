import json

import streamlit as st
import os
import PyPDF2 as pdf

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def get_gemini_response(prompt,pdf,input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt,pdf[0],input])
    return response.text

def input_pdfs_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text=""
    for page in reader.pages:
        text += page.extract_text()
    return text

#prompt template

input_prompt = """
Hey Act like a skilled or very exprienced ATS(Application Tracking system) with a deep understanding of tech field,
software engineering, data sciencem data analyst and big data engineer. your task is to evaluate the resume based on the given
job desciption and you must consider job market is very competitive and you should provide best assisstance for improving 
resumes. Assign the percentage Matching based on Job and 
the missing keywords with high accuracy:
resume:{text}
desciption:{jd}

I want the response in one string having the structure 
{{'JD Match':'%','Missing Keywords:[]','profile Summary':''}}


"""

st.title("Smart ATS")
st.text("Improve your resume ATS")
# st.set_page_config(page_title="ATS Resume EXpert")
# st.header("ATS Tracking System")
input_text=st.text_area("Job Description: ",key="input")
uploaded_file=st.file_uploader("Upload your resume(PDF)...",type=["pdf"],help="Upload a PDF file please")

submit = st.button("Submit")

if submit:
    if uploaded_file:
        text = input_pdfs_text(uploaded_file)
        response = get_gemini_response(input_prompt,text,input_text)
        st.subheader(response)

