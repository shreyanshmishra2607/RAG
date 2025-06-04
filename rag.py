import re
import string
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from serpapi import GoogleSearch
import os


# 1. Clean Text (for query & snippets)
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# 2. Query Classification
def classify_query(query):
    cleaned = clean_text(query)
    prompt = f"""
Classify the user query below. Reply only with one label from:
real-time, static, code, math, search, tool:calendar, tool:news

Query: "{cleaned}"
"""
    res = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    return res.json()["response"].strip().lower()

# 3. Search Web via SearxNG
SERP_API_KEY = os.getenv("SERP_API_KEY")

def search_web(query, max_results=5):
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "num": max_results
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    texts = []
    for r in results.get("organic_results", [])[:max_results]:
        snippet = r.get("snippet") or r.get("title")
        if snippet:
            texts.append(snippet)
    return texts

# 4. Chunk Text (split into max 100 words chunks)
def chunk_text(text, max_words=100):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# 5. Setup embedding model and ChromaDB collection
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection("my_knowledge")

# 6. Ingest chunks into ChromaDB
def ingest_chunks(chunks):
    embeddings = embed_model.encode(chunks).tolist()
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"doc_{i}"]
        )

# 7. Vector search
def vector_search(query, k=3):
    cleaned_query = clean_text(query)
    query_emb = embed_model.encode([cleaned_query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        include=["documents", "distances"]
    )
    docs = results["documents"][0]
    dists = results["distances"][0]

    print("\n🔍 Top matching chunks:")
    for i, doc in enumerate(docs):
        print(f"\nMatch {i+1} (dist {dists[i]:.4f}): {doc}")

    return "\n\n".join(docs)

# 8. Generate answer using LLM with context
def generate_answer(query, context):
    prompt = f"""Answer the question using the context below.

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

# 9. Main driver function
def main():
    user_input = input("Enter your query: ")
    print(f"Processing query: '{user_input}'")

    label = classify_query(user_input)
    print(f"Classification result: {label}")

    if label in ["static", "real-time", "code", "math"]:
        # Clear old docs before new ingest
        collection.delete(where={"$all": True})  # <-- Fix here

        # Get dynamic docs from web search
        web_texts = search_web(user_input)
        if not web_texts:
            print("No relevant web content found.")
            return

        # Clean + chunk all web results
        all_chunks = []
        for text in web_texts:
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)

        # Ingest chunks into vector store
        ingest_chunks(all_chunks)

        # Retrieve context & generate answer
        context = vector_search(user_input)
        answer = generate_answer(user_input, context)

        print("\n🗣️ Final Answer:\n", answer)
    else:
        print(f"Query type '{label}' is not supported for RAG flow.")

if __name__ == "__main__":
    main()
