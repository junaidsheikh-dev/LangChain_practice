from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="tell me a joke on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain this joke \n {joke}",
    input_variables= ['joke']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({'topic': 'cricket'})

print(result)