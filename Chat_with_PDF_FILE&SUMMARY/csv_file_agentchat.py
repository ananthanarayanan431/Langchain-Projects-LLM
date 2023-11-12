
from constant import kamalesh_oepnai,GOOGLE_API_KEY
import os

os.environ['OPENAI_API_KEY']=kamalesh_oepnai
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

import streamlit as st
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI

def main():
    st.set_page_config(page_title="Chat with CSV file")
    st.header("ASk anything from CSV file")

    csv_file = st.file_uploader("Upload a CSV file",type="csv")

    if csv_file:
        agent = create_csv_agent(
            OpenAI(temperature=0.9),
            csv_file,
            verbose=True
        )

        user_question =st.text_input("Now Ask question from CSV file")

        if user_question:
            with st.spinner(text="In Process..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()
