
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

from constant import openai_key

os.environ['OPENAI_API_KEY']= openai_key

loader = YoutubeLoader.from_youtube_url("https://youtu.be/Kn7SX2Mx_Jk?si=x9Pbv5J3Y5GS7e0X",add_video_info=True)

result = loader.load()

print(type(result)) #list

print(f"Author name {result[0].metadata['author']} and length is {result[0].metadata['length']} seconds long")

print(result) #complete transcript

llm = OpenAI(temperature=0.8)

chain = load_summarize_chain(llm=llm,chain_type='stuff',verbose=True)

#short YouTube Video Summary
print(chain.run(result))

loader1 = YoutubeLoader.from_youtube_url("https://youtu.be/X2mbi6WLPnk?si=ar-9BSLWB14wSDBG",add_video_info=True)
result1 = loader1.load()

print(f"Author name {result1[0].metadata['author']} and length is {result1[0].metadata['length']} seconds long")

text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
texts = text_split.split_documents(result1)

print(len(texts))

chain = load_summarize_chain(llm=llm,chain_type="map_reduce",verbose=False)

print(chain.run(texts))

youtube_urls = ['https://youtu.be/h_EbSqu8KI4?si=fvaoQ0wDlo2bXfu_','https://youtu.be/X2mbi6WLPnk?si=ar-9BSLWB14wSDBG']
text_split = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text = []

for url in youtube_urls:
    loader2 = YoutubeLoader.from_youtube_url(url,add_video_info=True)
    result2 = loader2.load()
    print(f"Author name {result2[0].metadata['author']} and length is {result2[0].metadata['length']} seconds long")
    text.extend(text_split.split_documents(result2))

print(chain.run(text))

