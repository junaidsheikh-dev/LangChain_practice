from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-20b", temperature=0)

result = model.invoke("what is the capital of pakistan")

print(result.content)