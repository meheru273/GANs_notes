from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template = ChatMessagePromptTemplate([
    ('system', SystemMessage(content="You are a helpful assistant.")),
    ('human', HumanMessage(content="{input}")),
])

prompt = chat_template.invoke({"input": "What is the capital of France?"})
print(prompt.content)
