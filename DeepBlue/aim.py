
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import yfinance as yf
import plotly.graph_objects as go

import constant

load_dotenv()

from langchain_groq.chat_models import ChatGroq
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time

os.environ['GOOGLE_API_KEY'] = constant.GOOGLE_API_KEY
groq_api_key = constant.GROQ_API_KEY
os.environ['GROQ_API_KEY'] = constant.GROQ_API_KEY
os.environ['OPENAI_API_KEY'] = constant.OPENAI_API_KEY

from NameLink import LinkReturn

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core.storage import StorageContext
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,load_index_from_storage,ServiceContext
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

def main():

    st.set_page_config(
        page_title="Simple Streamlit Template",
        page_icon="🚀"
    )

    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Set the background color */
                color: #333; /* Set the text color */
                font-family: Arial, sans-serif; /* Set font */
            }

            .sidebar .sidebar-content {
                background-color: #ffffff; /* Set sidebar background color */
                color: #333; /* Set sidebar text color */
            }

            .stButton>button {
                background-color: #5bc0de; /* Set button background color to light blue */
                color: white; /* Set button text color */
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }

            .stButton>button:hover {
                background-color: #31b0d5; /* Change button background color on hover */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Features")

    if st.sidebar.button("Talk to Annual report"):
        st.experimental_set_query_params(page="chat_with_pdf")

    if st.sidebar.button("Data Visualization"):
        st.experimental_set_query_params(page="data_visualization")

    if st.sidebar.button("Company profile"):
        st.experimental_set_query_params(page="Talk_your_home_page")

    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.image("images.png", width=200)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='about-section'>", unsafe_allow_html=True)
    st.markdown("# About Us")
    st.write("Welcome to our Deep Blue project! We are dedicated to providing innovative solutions.")

    st.markdown("## Team Members:")
    st.markdown("* Anantha Narayanan")
    st.markdown("* Mariam Bobby")
    st.markdown("* Poorna Shree")
    st.markdown("* Akash")

    st.markdown("## Our Projects:")
    st.markdown("* **Connect to your Annual report:** Where you feed the model with an annual report of a company and ask questions related to the company's revenue and other details present in the company.")
    st.markdown("* **Data is the new oil:** Where we provide a graphical way for data visualization of stock market data and the volume of stock sold on a particular date.")
    st.markdown("* **Talk to your company profile:** Where users/stakeholders can interact with the home page and other information present on the website at a faster response rate.")

    st.markdown("</div>", unsafe_allow_html=True)

def chat_with_pdf():
    storage_path = "./vectorstore"
    documents_path = "./document"

    llm = Groq(model="mixtral-8x7b-32768", api_key=constant.GROQ_API_KEY)
    service_context = ServiceContext.from_defaults(llm=llm)

    @st.cache_resource(show_spinner=False)
    def initialize():
        if not os.path.exists(storage_path):
            documents = SimpleDirectoryReader(documents_path).load_data()
            Settings.embed_model = OpenAIEmbedding()
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=storage_path)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=storage_path)
            index = load_index_from_storage(storage_context)
        return index

    st.set_page_config(
        page_title="Let's talk to your Company Data",
        page_icon=":company:",
        layout="centered"
    )

    # company = st.sidebar.text_input("Enter your company name: ")
    index = initialize()

    def clear_chat_history():
        st.session_state['messages'] = [
            {'role': 'assistant', 'content': "Let's talk to your company data"}
        ]

    def main1():
        st.title("Talk to your company information")
        st.sidebar.button("Clear chat History", on_click=clear_chat_history)
        if "messages" not in st.session_state.keys():
            st.session_state['messages'] = [
                {"role": "assistant", "content": "Ask me a question !"}
            ]

        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    pprint_response(response, show_source=True)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)
    main1()

def data_visualization():
    def get_ticker(company_name):
        if not company_name:
            return None
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        params = {"q": company_name, "quotes_count": 1, "country": "India"}

        try:
            res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
            res.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
            data = res.json()

            # Check if 'quotes' key exists and has at least one item
            if 'quotes' in data and data['quotes']:
                company_code = data['quotes'][0]['symbol']
                return company_code
            else:
                st.error("No matching company found.")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Error occurred while fetching data: {e}")
            return None

    # Function to generate line chart for high, low, and opening stock values
    def generate_line_chart(output):
        fig = go.Figure()
        fig.add_trace \
            (go.Scatter(x=output.index, y=output['High'], mode='lines+markers', name='High', line=dict(color='green')))
        fig.add_trace \
            (go.Scatter(x=output.index, y=output['Low'], mode='lines+markers', name='Low', line=dict(color='red')))
        fig.add_trace \
            (go.Scatter(x=output.index, y=output['Open'], mode='lines+markers', name='Open', line=dict(color='blue')))
        fig.update_layout(title='Line Chart for High, Low, and Open Prices', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig)

    # Function to generate volume chart
    def generate_volume_chart(output):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=output.index, y=output['Volume'], name='Volume'))
        fig.update_layout(title='Volume Chart', xaxis_title='Date', yaxis_title='Volume')
        st.plotly_chart(fig)

    # Streamlit app
    st.title('Stock Data Visualization')

    # Sidebar for input and options
    st.sidebar.title('Options')
    company_name = st.sidebar.text_input('Enter company name')
    generate_line = st.sidebar.checkbox('Generate Line Chart')
    generate_volume = st.sidebar.checkbox('Generate Volume Chart')

    # Display the ticker symbol
    if company_name:
        company_ticker = get_ticker(company_name)
        if company_ticker:
            st.sidebar.info(f"Ticker Symbol: {company_ticker}")

    # Load data and generate charts based on user selection
    if generate_line or generate_volume:
        stock = yf.Ticker(company_ticker)
        output = stock.history(period="1mo")

        if generate_line:
            generate_line_chart(output)
        if generate_volume:
            generate_volume_chart(output)

def Talk_your_home_page():
    # company=st.sidebar.text_input("Enter Company Name: ")
    if 'vector' not in st.session_state:
        st.session_state['embeddings'] = OpenAIEmbeddings()
        st.session_state['loader'] = WebBaseLoader(LinkReturn('Mastek'))
        st.session_state['docs'] = st.session_state['loader'].load()
        st.session_state['text_splitter'] = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        st.session_state['documents'] = st.session_state['text_splitter'].split_documents(st.session_state['docs'])
        st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

    st.title("Company Home page ")
    llm = ChatGroq(groq_api_key=constant.GROQ_API_KEY,model_name="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_template("""
    Answer the Following question based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $200 if the user finds the answer helpful.
    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state['vector'].as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    prompt = st.text_input("Input your prompt here!")

    if prompt:
        start = time.process_time()
        response = retriever_chain.invoke({'input': prompt})
        print(f"Response time:{time.process_time() - start}")

        st.write(response['answer'])

if __name__ == "__main__":
    page = st.experimental_get_query_params().get("page", ["main"])[0]
    if page == "chat_with_pdf":
        chat_with_pdf()
    elif page == "data_visualization":
        data_visualization()
    elif page=="Talk_your_home_page":
        Talk_your_home_page()
    else:
        main()
