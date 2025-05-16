from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
#chat template 
chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ('human', MessagesPlaceholder(variable_name="input")),
])

history = []

#load chat 
with open("history.txt") as f:
    history.extend(f.readlines())
    
#create propmpt
chat_template.invoke({'history': history , 'input': "What is the capital of France?"})