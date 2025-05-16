from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0.7, max_tokens=1000)  

"""Temparature is a parameter that controls the randomness of the model's output.
if it is between 0.0-0.3 then deterministic and less random 
if it is 0.7-1.5 then more random and creative"""
""" if themparature is 0.0 then the model will always give the same output for the same input"""
"""max_tokens is the maximum number of tokens in the output."""

result=model.invoke("What is the capital of France?")

print(result)