from langchain_google_genai import ChatGoogleGenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenAI(model="gemini-1.5-turbo", temperature=0.7, max_tokens=1000)
result = model.invoke("What is the capital of France?")   

print(result.content)