from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()


model = ChatGroq(model = "openai/gpt-oss-20b")

prompt1 = PromptTemplate(
    template="write a detail note on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "sumarize the following text in paragraph format \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

text_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 200, RunnableSequence(prompt2 , model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(text_gen_chain, branch_chain)

result = chain.invoke({'topic': 'hill station'})

print(result)