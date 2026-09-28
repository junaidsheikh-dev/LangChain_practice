from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

prompt1 = PromptTemplate(
    template="Genrate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="genrate 5 short question answers from the following text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="merge the following notes and questions in single document \n notes -> {notes} \n questions -> {questions}",
    input_variables=['notes', 'questions']
)


model = ChatGroq(model = "openai/gpt-oss-20b", temperature=0)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes = prompt1 | model | parser ,
    questions = prompt2 | model | parser
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

document = """Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks that normally require human intelligence. These tasks include understanding language, recognizing images, making decisions, solving problems, and learning from experience.

Machine Learning (ML) is a subset of artificial intelligence. Instead of explicitly programming a computer with every rule needed to solve a problem, machine learning allows computers to learn patterns from data. A machine learning model is trained using data and can then use what it has learned to make predictions or decisions about new data.

There are three common types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, a model learns from labeled data, where the correct answer is already known. For example, a model can learn to identify spam emails by training on emails that have already been labeled as spam or not spam.

In unsupervised learning, the data does not contain predefined labels. The algorithm attempts to discover patterns or groups within the data. Customer segmentation is an example where an algorithm might group customers based on their purchasing behavior.

Reinforcement learning works differently. An agent interacts with an environment and learns by receiving rewards or penalties for its actions. Over time, the agent learns which actions are more likely to produce desirable results. Reinforcement learning is commonly associated with robotics, games, and decision-making systems.

Training a machine learning model involves providing data and adjusting the model's parameters so that its predictions become more accurate. After training, the model should be tested using data it has not seen before. This helps determine whether the model can generalize its knowledge to new situations.

AI and ML are used in many areas, including healthcare, finance, transportation, education, cybersecurity, and entertainment. However, these technologies also introduce challenges such as data privacy, security, bias, and the need for responsible use."""

result = chain.invoke(document)

print(result)

chain.get_graph().print_ascii()