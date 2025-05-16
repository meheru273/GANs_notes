""" for ambigious queries
    generates multiple queries from user query"""
    
from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings 
from langchain_core.documents import Document 
from langchain_openai import ChatOpenAI 
from langchain.retrievers.multi_query import MultiQueryRetriever 

documents = [
    Document(page_content="Langchain is a tool to develop LLM applications"),
    Document(page_content="OpenAI API provides ways to use openai models")   
]

#initialze openai model
embedding_model = OpenAIEmbeddings()

#create FAISS vectore store 
vectorstore = FAISS.from_documents(documents=documents,embedding_model=embedding_model)

similarity_retriever = vectorstore.as_retriever(search_type='similarity',search_kwargs={'k':2})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever = vectorstore.as_retriever(search_kwargs={"k":5}),
    llm= ChatOpenAI(model="gpt-3.5-turbo")
)

query = "what is ai?"

similarity_res = similarity_retriever.invoke(query)
multiquery_res = multiquery_retriever.invoke(query)