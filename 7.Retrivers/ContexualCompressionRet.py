""" Contextual Compression Retriever"""

from langchain_community.vectorstores import FAISS 
from langchain_openai import OpenAIEmbeddings 
from langchain_core.documents import Document 
from langchain_openai import ChatOpenAI 
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

documents = [
    Document(page_content="Langchain is a tool to develop LLM applications"),
    Document(page_content="OpenAI API provides ways to use openai models")   
]

#create a FAISS  vector store from documents

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=documents,embedding_model=embedding_model)

base_retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# set up compressor using llm 
llm= ChatOpenAI()
compressor = LLMChainExtractor.from_llm(llm)

#create the contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
) 

query = "what is ai?"

result = compression_retriever.invoke(query)
