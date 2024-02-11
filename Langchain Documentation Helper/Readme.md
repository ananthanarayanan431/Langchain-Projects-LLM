
  mkdir langchain-docs

  
  wget -r -A.html -P langchain-docs  https://api.python.langchain.com/en/latest


 use the above code to download the langchain documents in html format. GenAI don't care about the format of content
  and copy the path paste it in ReadTheDocsLoader() in ingestion.py file


  we're using free version of Pinecone vector database and Google's Gemini-pro model
