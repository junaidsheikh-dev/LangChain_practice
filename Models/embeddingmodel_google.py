from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

model = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001", output_dimensionality=32)

documents = [
    "Python is a popular programming language used for web development, data science, and automation.",
    "FastAPI is a modern Python framework for building high-performance web APIs.",
    "LangChain is a framework for building applications powered by large language models.",
    "Embeddings convert text into numerical vectors that represent semantic meaning.",
    "PostgreSQL is a relational database commonly used for storing structured application data.",
    "Machine learning allows computers to learn patterns from data.",
    "Artificial intelligence enables computers to perform tasks that normally require human intelligence.",
    "The weather is sunny and warm today.",
    "I enjoy playing football with my friends on weekends.",
    "Pizza is one of my favorite foods.",
]

doc_embedding = model.embed_documents(documents)

query = "How can I build an API using Python?"
query_embedding = model.embed_query(query)



scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Query: {query}")
print(documents[index])

print(f"Similarity Score: {score}")