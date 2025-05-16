from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def word_count(text: str) -> int:
    return len(text.split())

# 1. Define the “facts” serial chain (prompt → model → parser)
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"],
)
model = ChatOpenAI()
parser = StrOutputParser()
facts_chain = RunnableSequence([prompt, model, parser])

# 2. Build the parallel stage
parallel = RunnableParallel({
    # a) keep the facts string
    "facts": facts_chain,
    # b) run word_count *after* we have the facts
    "word_count": RunnableSequence([facts_chain, RunnableLambda(word_count)]),
})

# 3. Merge into one final string
merge = RunnableLambda(lambda d: f"Word count: {d['word_count']}, Facts: {d['facts']}")

# 4. Compose: serial → parallel → merge
chain = RunnableSequence([parallel, merge])

# 5. Invoke
result = chain.invoke({"topic": "Python programming"})
print(result)
