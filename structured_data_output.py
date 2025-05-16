from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

model = ChatOpenAI()

#schema 
class Review(TypedDict):
    summary : Annotated[str,"a short summary of the text"]
    sentiment : Annotated[str,"negative,positive,neutral"]
    
structured_model = model.with_structured_output(Review)
    

result = structured_model.invoke(""" sm,d ncierjovemrf,  df vkerm fkvc
                      ds,c kefemrkf gfdm vkr vfrk sfder""")
