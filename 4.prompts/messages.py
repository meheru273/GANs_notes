from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_tokens=1000)
messages = [
    SystemMessage(content = " yo bitch wassap"),
    HumanMessage(content = "tell  me about langchai")
    
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
