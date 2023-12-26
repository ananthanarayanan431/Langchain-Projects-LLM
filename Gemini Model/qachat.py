
import streamlit as st
from constant import Gemini_API
import os

os.environ['GOOGLE_API_KEY']= Gemini_API

import google.generativeai as genai

genai.configure(api_key=Gemini_API)

model=genai.GenerativeModel("gemini-pro")

chat = model.start_chat(history=[])

def get_response(question)->str:
    response = chat.send_message(question,stream=True)
    return response

st.set_page_config(page_title="Q-A Demo")

st.header("Gemini LLM Application")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

input1 = st.text_input("Input ",key="input")
submit = st.button("Ask the question")

if submit and input1:
    response = get_response(input1)

    st.session_state['chat_history'].append(('you',input1))
    st.subheader("The Response is: ")
    print(response)
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(('Bot',chunk.text))

    # st.session_state['chat_history'].append(('Bot',response))

st.subheader("The Chat History is")

for role,text in st.session_state['chat_history']:
    st.write(f"{role} : {text}")

