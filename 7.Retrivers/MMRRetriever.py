from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document 

documents = [
    Document(page_content="Langchain is a tool to develop LLM applications"),
    Document(page_content="OpenAI API provides ways to use openai models")
    
]

#initialize embedding model
embedding_model = OpenAIEmbeddings()

 # create FAISS vector store from documents
vectorstore = FAISS.from_documents(
     documents=documents,
     embedding=embedding_model,
 )

#enable MMR in retriever
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs={"k":3,"lambda_mult":1}
    
    #k= top results
    #lambda_mult = relevance diversity balance"""
)

query = "what is langchain"

results = retriever.invoke(query)

