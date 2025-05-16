from langchain_community.retrievers import WikipediaRetriever 


retriever = WikipediaRetriever(top_k_results=2,lang='en')

query = "the geopolitical history of bangladesh"

docs = retriever.invoke(query)

""" retriever is a runnable class"""

for i,doc in enumerate(docs):
    print({i+1})
    print(doc.page_content)