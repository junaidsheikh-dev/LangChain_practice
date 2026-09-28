from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

class Feedback(BaseModel):
    sentiments : Literal['positive', 'negative'] = Field(description='give the sentiment of the feedback')


model_google = ChatGoogleGenerativeAI(model ="gemini-3.8-flash")
model_groq = ChatGroq(model = "openai/gpt-oss-20b")

parser = StrOutputParser()
pydentic_parser = PydanticOutputParser(pydantic_object= Feedback)

prompt1 = PromptTemplate(
    template= "classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format}",
    input_variables= ['feedback'],
    partial_variables={'format': pydentic_parser.get_format_instructions()}
)


classifier_chain = prompt1 | model_groq | pydentic_parser

prompt2 = PromptTemplate(
    template = "write an appropriate response in a paragraph to this positive feedback \n {feedback}",
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = "write an appropriate response in a paragraph to this negative feedback \n {feedback}",
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiments == 'positive', prompt2 | model_groq | parser),
    (lambda x:x.sentiments == 'negative', prompt3 | model_groq | parser),
    RunnableLambda(lambda x: "could not find sentiments")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback" : "this is not bad phone"})

print(result)