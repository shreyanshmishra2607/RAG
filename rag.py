import re
import string
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))  
    return text

def classify_query(query: str) -> str:
    cleaned = clean_text(query)
    prompt = f"""
Classify the following user query. Does it need real-time (latest news, updates) or static (general facts) info? 

Or is it related to code, math, search, or tools like calendar or news?

Query: "{cleaned}"

Reply only with one of these labels:
- real-time
- static
- code
- math
- search
- tool:calendar
- tool:news
"""
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"].strip().lower()

# if __name__ == "__main__":
#     user_input = input("Enter your query: ")
#     print(f"Processing query: '{user_input}'")
#     label = classify_query(user_input)
#     print("Classification result:", label)  

# embedding, and storing in ChromaDB

# Load embedding model
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# Setup Chroma DB
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",  # where to save
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection("my_knowledge")

# Function: Chunking (very basic for now)
def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# Input text
document = """
Artificial Intelligence (AI) is revolutionizing the world. From chatbots to self-driving cars, it's changing industries at a rapid pace...
"""  # You can load from file or web

# 1. Chunk
chunks = chunk_text(document)

# 2. Embed
embeddings = embed_model.encode(chunks).tolist()

# 3. Store in Chroma
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        embeddings=[embeddings[i]],
        ids=[f"doc_{i}"]
    )

print("Chunks embedded and stored in ChromaDB ✅")

# searching
def vector_search(query: str, k: int = 3):
    # Clean the query
    cleaned_query = clean_text(query)

    # Embed the query
    query_embedding = embed_model.encode([cleaned_query]).tolist()[0]

    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    # Extract the top matching documents
    documents = results["documents"][0]
    distances = results["distances"][0]

    print("\n🔍 Top Matches:")
    for i in range(len(documents)):
        print(f"\nMatch {i+1} (Distance: {distances[i]:.4f}):\n{documents[i]}")

    # Combine context for generation
    combined_context = "\n\n".join(documents)
    return combined_context

def generate_answer(query, context):
    prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question:
{query}

Answer:"""

    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    return res.json()["response"].strip()


def main():
    user_input = input("Enter your query: ")
    print(f"Processing query: '{user_input}'")

    label = classify_query(user_input)
    print(f"Classification result: {label}")

    if label in ["static", "real-time", "code", "math"]:
        context = vector_search(user_input, k=3)
        print("\n🧠 Context Retrieved:\n", context)

        answer = generate_answer(user_input, context)
        print("\n🗣️ Final Answer:\n", answer)

if __name__ == "__main__":
    main()

