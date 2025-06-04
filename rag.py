import re
import string
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

Examples:
- "what is python" -> static
- "how to code in python" -> code  
- "what is 2+2" -> math
- "current weather" -> real-time
- "news today" -> tool:news

Query: "{cleaned}"
Classification:"""
    
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        
        if res.status_code == 200:
            classification = res.json()["response"].strip().lower()
            # Clean up the response to extract just the label
            for label in ["real-time", "static", "code", "math", "search", "tool:calendar", "tool:news"]:
                if label in classification:
                    return label
            return "static"  # default fallback
        else:
            print(f"LLM API error: {res.status_code}")
            return "static"  # fallback
    except Exception as e:
        print(f"Classification error: {e}")
        return "static"  # fallback

# 3. Search Web - Multiple options
SERP_API_KEY = os.getenv("SERP_API_KEY")

def search_web_serpapi(query, max_results=5):
    """Search using SerpAPI"""
    if not SERP_API_KEY:
        return []
    
    try:
        params = {
            "q": query,
            "api_key": SERP_API_KEY,
            "engine": "google",
            "num": max_results,
            "gl": "us",
            "hl": "en"
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            print(f"SerpAPI Error: {results['error']}")
            return []

        texts = []
        organic_results = results.get("organic_results", [])
        
        for r in organic_results[:max_results]:
            snippet = r.get("snippet", "") or r.get("title", "")
            if snippet:
                texts.append(snippet)
                
        return texts
        
    except Exception as e:
        print(f"SerpAPI error: {e}")
        return []

def search_web_duckduckgo(query, max_results=5):
    """Fallback search using DuckDuckGo (requires duckduckgo-search package)"""
    try:
        from duckduckgo_search import DDGS
        
        texts = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                snippet = r.get('body', '') or r.get('title', '')
                if snippet:
                    texts.append(snippet)
        return texts
    except ImportError:
        print("duckduckgo-search package not installed. Install with: pip install duckduckgo-search")
        return []
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

def search_web_mock(query, max_results=5):
    """Mock search with predefined content for testing"""
    mock_data = {
        "python": [
            "Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
            "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics developed by Guido van Rossum. It was originally released in 1991.",
            "Python emphasizes code readability with the use of significant indentation. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.",
            "Python is dynamically-typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
            "Python's design philosophy emphasizes code readability and a syntax that allows programmers to express concepts in fewer lines of code than might be used in languages such as C++ or Java."
        ],
        "machine learning": [
            "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to learn and make decisions from data.",
            "Machine learning algorithms build mathematical models based on training data in order to make predictions or decisions without being explicitly programmed to do so.",
            "Machine learning is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning."
        ]
    }
    
    # Simple keyword matching
    query_lower = query.lower()
    for key, content in mock_data.items():
        if key in query_lower:
            return content[:max_results]
    
    # Default generic content
    return [
        f"This is mock search result for '{query}'. In a real implementation, this would contain relevant web content.",
        f"Another mock result about '{query}' with relevant information that would help answer the user's question.",
        f"Third mock result providing additional context about '{query}' for comprehensive understanding."
    ]

def search_web(query, max_results=5):
    """Main search function with fallback options"""
    print(f"🔍 Searching for: '{query}'")
    
    # Try SerpAPI first
    if SERP_API_KEY:
        print("  Using SerpAPI...")
        results = search_web_serpapi(query, max_results)
        if results:
            print(f"✅ Found {len(results)} results from SerpAPI")
            return results
    
    # Try DuckDuckGo as fallback
    print("  Trying DuckDuckGo...")
    results = search_web_duckduckgo(query, max_results)
    if results:
        print(f"✅ Found {len(results)} results from DuckDuckGo")
        return results
    
    # Use mock data as last resort
    print("  Using mock data for testing...")
    results = search_web_mock(query, max_results)
    print(f"✅ Using {len(results)} mock results")
    return results

# 4. Chunk Text (split into max 100 words chunks)
def chunk_text(text, max_words=100):
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if chunk.strip():  # only add non-empty chunks
            chunks.append(chunk)
    return chunks

# 5. Setup embedding model and ChromaDB collection
try:
    embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    print("✅ Embedding model loaded successfully")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Falling back to a different model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    chroma_client = chromadb.Client(Settings(
        persist_directory="./chroma_store",
        anonymized_telemetry=False
    ))
    collection = chroma_client.get_or_create_collection("my_knowledge")
    print("✅ ChromaDB initialized successfully")
except Exception as e:
    print(f"ChromaDB initialization error: {e}")
    exit(1)

# 6. Ingest chunks into ChromaDB
def ingest_chunks(chunks):
    if not chunks:
        print("No chunks to ingest")
        return
        
    try:
        # Clear existing documents
        try:
            # Get all existing IDs and delete them
            existing = collection.get()
            if existing['ids']:
                collection.delete(ids=existing['ids'])
                print(f"Cleared {len(existing['ids'])} existing documents")
        except Exception as e:
            print(f"Warning: Could not clear existing documents: {e}")
        
        embeddings = embed_model.encode(chunks).tolist()
        
        ids = []
        documents = []
        emb_list = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"doc_{i}")
            documents.append(chunk)
            emb_list.append(embeddings[i])
        
        collection.add(
            documents=documents,
            embeddings=emb_list,
            ids=ids
        )
        print(f"✅ Ingested {len(chunks)} chunks into vector store")
        
    except Exception as e:
        print(f"Error ingesting chunks: {e}")

# 7. Vector search
def vector_search(query, k=3):
    try:
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
            print(f"\nMatch {i+1} (dist {dists[i]:.4f}): {doc[:200]}...")

        return "\n\n".join(docs)
    except Exception as e:
        print(f"Vector search error: {e}")
        return ""

# 8. Generate answer using LLM with context
def generate_answer(query, context):
    if not context.strip():
        return "I couldn't find relevant information to answer your question."
        
    prompt = f"""Answer the question using the context below. Be concise and helpful.

Context:
{context}

Question:
{query}

Answer:"""
    
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        
        if res.status_code == 200:
            return res.json()["response"].strip()
        else:
            return f"Error generating answer: HTTP {res.status_code}"
    except Exception as e:
        return f"Error generating answer: {e}"

# 9. Main driver function
def main():
    user_input = input("Enter your query: ")
    print(f"Processing query: '{user_input}'")

    label = classify_query(user_input)
    print(f"Classification result: {label}")

    if label in ["static", "real-time", "code", "math"]:
        # Get dynamic docs from web search
        print("🔍 Searching the web...")
        web_texts = search_web(user_input)
        
        if not web_texts:
            print("❌ No relevant web content found.")
            print("This could be due to:")
            print("1. Missing or invalid SERP_API_KEY")
            print("2. Network connectivity issues")
            print("3. API quota exceeded")
            print("4. Query returned no results")
            return

        # Clean + chunk all web results
        print("📝 Processing and chunking web content...")
        all_chunks = []
        for i, text in enumerate(web_texts):
            cleaned = clean_text(text)
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks)
            print(f"  Text {i+1}: {len(chunks)} chunks")

        if not all_chunks:
            print("❌ No valid chunks created from web content")
            return

        # Ingest chunks into vector store
        print("💾 Ingesting into vector store...")
        ingest_chunks(all_chunks)

        # Retrieve context & generate answer
        print("🔎 Performing vector search...")
        context = vector_search(user_input)
        
        if context:
            print("🤖 Generating answer...")
            answer = generate_answer(user_input, context)
            print("\n" + "="*50)
            print("🗣️ Final Answer:")
            print("="*50)
            print(answer)
        else:
            print("❌ No relevant context found")
    else:
        print(f"Query type '{label}' is not supported for RAG flow.")

if __name__ == "__main__":
    main()