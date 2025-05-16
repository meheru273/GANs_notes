from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=1000) 

chat_history = [
    SystemMessage(content="yo")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break
    
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print(f"Chatbot: {result.content}")
    