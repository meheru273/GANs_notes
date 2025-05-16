from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-2", temperature=0.7, max_tokens=1000)

result = model.invoke("What is the capital of France?")

print(result.content)

"chat models produces losts of additional informations , using 'content, ' to get the main content"

