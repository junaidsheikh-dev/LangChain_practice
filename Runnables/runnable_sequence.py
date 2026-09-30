from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="tell me a joke on {topic}",
    input_variables=['topic']
)


parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser)

result = chain.invoke({'topic': 'school'})

print(result)