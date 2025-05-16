from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document 

documents = [
    Document(page_content="Langchain is a tool to develop LLM applications"),
    Document(page_content="OpenAI API provides ways to use openai models")
    
]

#initialize embedding
embedding_model= OpenAIEmbeddings()

#Create Chroma vector
vectorsotre = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name='my_store'
)

#convert into retriever 
retriever = vectorsotre.as_retriever(search_kwargs={"k":2})

query = "what is OpenAI"
results = retriever.invoke(query)

""" more advanced than vector store and can be formed chains"""