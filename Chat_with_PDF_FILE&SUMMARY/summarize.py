
import os
from constant import openai_key

os.environ['OPENAI_API_KEY']= openai_key

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import gradio as gr
import tiktoken
import openai

loader = PyPDFLoader("Fine-Tune Your Own Llama 2 Model in a Colab Notebook _ Towards Data Science.pdf")

doc =loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

texts = text_splitter.split_documents(doc)

# print(len(texts))

llm = OpenAI(temperature=0.5)

chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    verbose=True
)
# print(chain.run(texts))

def summarize(pdf_path):
    loader1 = PyPDFLoader(pdf_path)
    doc1 = loader1.load()
    text_splitter1 = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    text1 = text_splitter1.split_text(doc1)


    chain1 = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        verbose=True
    )

    return chain1.run(text1)

input_path = gr.components.Textbox(label="Provide PDF file path")
output_summary = gr.components.Textbox(label="Summary")

interface = gr.Interface(
    fn=summarize,
    inputs=input_path,
    outputs=output_summary,
    title="PDF Summarizer",
    description="Provide the PDF File path to get the Summary"
).launch(share=True)
