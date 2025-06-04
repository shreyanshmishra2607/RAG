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

# Configuration from environment variables
SERP_API_KEY = os.getenv("SERP_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

print(f"🔧 Configuration loaded:")
print(f"   SERP_API_KEY: {'✅ Set' if SERP_API_KEY else '❌ Missing'}")
print(f"   OLLAMA_HOST: {OLLAMA_HOST}")
print(f"   OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"   OLLAMA_TIMEOUT: {OLLAMA_TIMEOUT}s")

# 1. Clean Text (for query & snippets)
def clean_text(text):
    """Clean and normalize text for better processing"""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# 2. Query Classification
def classify_query(query):
    """Classify user query to determine processing approach"""
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
        print("🤖 Classifying query...")
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=OLLAMA_TIMEOUT)
        
        if res.status_code == 200:
            classification = res.json()["response"].strip().lower()
            # Clean up the response to extract just the label
            for label in ["real-time", "static", "code", "math", "search", "tool:calendar", "tool:news"]:
                if label in classification:
                    print(f"✅ Query classified as: {label}")
                    return label
            print("⚠️ Could not determine classification, defaulting to 'static'")
            return "static"
        else:
            print(f"❌ LLM API error: {res.status_code}")
            return "static"
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Is it running?")
        return "static"
    except requests.exceptions.Timeout:
        print(f"⏰ Ollama request timeout ({OLLAMA_TIMEOUT}s)")
        return "static"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "static"

# 3. Web Search Functions
def search_web_serpapi(query, max_results=5):
    """Search using SerpAPI (primary method)"""
    if not SERP_API_KEY:
        print("⚠️ SERP_API_KEY not found in environment")
        return []
    
    try:
        print(f"🔍 Searching with SerpAPI: '{query}'")
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
            print(f"❌ SerpAPI Error: {results['error']}")
            return []

        texts = []
        organic_results = results.get("organic_results", [])
        
        for i, r in enumerate(organic_results[:max_results]):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            
            # Combine title and snippet for richer context
            combined_text = f"{title}. {snippet}" if title and snippet else (title or snippet)
            
            if combined_text:
                texts.append(combined_text)
                print(f"   Result {i+1}: {combined_text[:100]}...")
                
        print(f"✅ Found {len(texts)} results from SerpAPI")
        return texts
        
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return []

def search_web_duckduckgo(query, max_results=5):
    """Fallback search using DuckDuckGo"""
    try:
        from duckduckgo_search import DDGS
        
        print(f"🦆 Searching with DuckDuckGo: '{query}'")
        texts = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for i, r in enumerate(results):
                title = r.get('title', '')
                body = r.get('body', '')
                
                # Combine title and body for richer context
                combined_text = f"{title}. {body}" if title and body else (title or body)
                
                if combined_text:
                    texts.append(combined_text)
                    print(f"   Result {i+1}: {combined_text[:100]}...")
        
        print(f"✅ Found {len(texts)} results from DuckDuckGo")
        return texts
    except ImportError:
        print("❌ duckduckgo-search package not installed. Install with: pip install duckduckgo-search")
        return []
    except Exception as e:
        print(f"❌ DuckDuckGo search error: {e}")
        return []

def search_web(query, max_results=5):
    """Main search function with fallback options"""
    print(f"\n🔍 Starting web search for: '{query}'")
    
    # Try SerpAPI first
    if SERP_API_KEY:
        results = search_web_serpapi(query, max_results)
        if results:
            return results
    
    # Try DuckDuckGo as fallback
    print("🔄 Trying DuckDuckGo as fallback...")
    results = search_web_duckduckgo(query, max_results)
    if results:
        return results
    
    # If all searches fail
    print("❌ All search methods failed. Please check:")
    print("   1. Internet connection")
    print("   2. SERP_API_KEY in .env file")
    print("   3. duckduckgo-search package installation")
    return []

# 4. Text Processing
def chunk_text(text, max_words=100):
    """Split text into manageable chunks for better embedding"""
    if not text or not text.strip():
        return []
    
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# 5. Setup embedding model and ChromaDB
def initialize_components():
    """Initialize embedding model and ChromaDB"""
    # Initialize embedding model
    try:
        print("🧠 Loading embedding model...")
        embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        print("✅ Nomic embedding model loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading Nomic model: {e}")
        print("🔄 Falling back to all-MiniLM-L6-v2...")
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Fallback embedding model loaded successfully")
        except Exception as e2:
            print(f"❌ Failed to load any embedding model: {e2}")
            exit(1)

    # Initialize ChromaDB
    try:
        print("🗄️ Initializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        collection = chroma_client.get_or_create_collection("dynamic_knowledge")
        print("✅ ChromaDB initialized successfully")
        return embed_model, collection
    except Exception as e:
        print(f"❌ ChromaDB initialization error: {e}")
        exit(1)

# Initialize components globally
embed_model, collection = initialize_components()

# 6. Vector Store Operations
def ingest_chunks(chunks):
    """Ingest text chunks into the vector store"""
    if not chunks:
        print("⚠️ No chunks to ingest")
        return False
        
    try:
        print(f"💾 Ingesting {len(chunks)} chunks into vector store...")
        
        # Clear existing documents for fresh search results
        try:
            existing = collection.get()
            if existing['ids']:
                collection.delete(ids=existing['ids'])
                print(f"🗑️ Cleared {len(existing['ids'])} existing documents")
        except Exception as e:
            print(f"⚠️ Could not clear existing documents: {e}")
        
        # Generate embeddings
        print("🔢 Generating embeddings...")
        embeddings = embed_model.encode(chunks).tolist()
        
        # Prepare data for insertion
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Add to collection
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks")
        return True
        
    except Exception as e:
        print(f"❌ Error ingesting chunks: {e}")
        return False

def vector_search(query, k=3):
    """Perform semantic search to find relevant chunks"""
    try:
        print(f"🔎 Performing vector search (top-{k})...")
        cleaned_query = clean_text(query)
        query_emb = embed_model.encode([cleaned_query]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "distances"]
        )
        
        docs = results["documents"][0]
        dists = results["distances"][0]

        if not docs:
            print("❌ No matching documents found")
            return ""

        print(f"\n📋 Top {len(docs)} matching chunks:")
        for i, (doc, dist) in enumerate(zip(docs, dists)):
            print(f"   Match {i+1} (similarity: {1-dist:.3f}): {doc[:150]}...")

        # Combine all relevant chunks
        context = "\n\n".join(docs)
        print(f"✅ Retrieved {len(docs)} chunks for context")
        return context
        
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        return ""

# 7. Answer Generation
def generate_answer(query, context):
    """Generate answer using LLM with retrieved context"""
    if not context or not context.strip():
        return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
        
    prompt = f"""Answer the question using the context below. Be concise, helpful, and accurate.

Context:
{context}

Question: {query}

Instructions:
- Use only information from the context above
- Be specific and factual
- If the context doesn't fully answer the question, say so
- Keep the answer concise but complete

Answer:"""
    
    try:
        print("✨ Generating answer with LLM...")
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=OLLAMA_TIMEOUT)
        
        if res.status_code == 200:
            answer = res.json()["response"].strip()
            print("✅ Answer generated successfully")
            return answer
        else:
            return f"Error generating answer: HTTP {res.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama service. Please ensure Ollama is running."
    except requests.exceptions.Timeout:
        return f"Error: Request timeout ({OLLAMA_TIMEOUT}s). Try a simpler question."
    except Exception as e:
        return f"Error generating answer: {e}"

# 8. Math Query Handler
def handle_math_query(query):
    """Handle direct mathematical calculations"""
    try:
        print("🧮 Attempting direct math calculation...")
        # Clean the query to extract mathematical expression
        math_expr = query.lower()
        math_expr = math_expr.replace("what is", "").replace("calculate", "")
        math_expr = math_expr.replace("compute", "").replace("solve", "")
        math_expr = math_expr.replace("×", "*").replace("÷", "/")
        math_expr = math_expr.replace("plus", "+").replace("minus", "-")
        math_expr = math_expr.strip()
        
        # Safety check - only allow basic math operations
        allowed_chars = set("0123456789+-*/.()\s")
        if all(c in allowed_chars for c in math_expr) and math_expr:
            result = eval(math_expr)
            print(f"✅ Direct calculation: {math_expr} = {result}")
            return f"The answer is: {result}"
        else:
            print("🔍 Complex math expression, will search for context...")
            return None
    except:
        print("⚠️ Could not compute directly, will search for context...")
        return None

# 9. Main Processing Function
def process_query(user_input):
    """Main function to process user queries"""
    print(f"\n{'='*60}")
    print(f"🚀 Processing query: '{user_input}'")
    print(f"{'='*60}")

    # Classify the query
    label = classify_query(user_input)
    
    # Handle math queries with direct calculation first
    if label == "math":
        direct_answer = handle_math_query(user_input)
        if direct_answer:
            return direct_answer

    # For all query types, get dynamic information from web
    print(f"\n📊 Query type: {label}")
    print("🌐 Fetching dynamic information from web...")

    # Search the web for current information
    web_texts = search_web(user_input)
    
    if not web_texts:
        return ("I couldn't find current information about your query. "
                "This might be due to network issues or API limitations. "
                "Please check your internet connection and API keys.")

    # Process and chunk the web content
    print(f"\n📝 Processing {len(web_texts)} web results...")
    all_chunks = []
    for i, text in enumerate(web_texts):
        if text.strip():  # Only process non-empty texts
            cleaned = clean_text(text)
            if cleaned:  # Only chunk if cleaning produced valid text
                chunks = chunk_text(text)  # Use original text, not cleaned
                all_chunks.extend(chunks)
                print(f"   Text {i+1}: Generated {len(chunks)} chunks")

    if not all_chunks:
        return "I found some web results but couldn't process them properly. Please try rephrasing your question."

    # Ingest into vector store
    if not ingest_chunks(all_chunks):
        return "There was an error processing the information. Please try again."

    # Retrieve relevant context
    context = vector_search(user_input, k=5)  # Get top 5 most relevant chunks
    
    if not context:
        return "I couldn't find relevant information in the search results. Please try a different question."

    # Generate final answer
    answer = generate_answer(user_input, context)
    return answer

# 10. Main Driver Function
def main():
    """Main interactive loop"""
    print("🤖 Dynamic RAG System Ready!")
    print("💡 This system searches the web in real-time for every query")
    print("🔄 Type 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("\n📝 Enter your query: ").strip()
            
            if not user_input:
                print("⚠️ Please enter a valid query")
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            # Process the query
            answer = process_query(user_input)
            
            # Display the final answer
            print(f"\n{'='*60}")
            print("🎯 FINAL ANSWER:")
            print(f"{'='*60}")
            print(answer)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    main()