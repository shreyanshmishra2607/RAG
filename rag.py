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
# Replace your incomplete search_web_serpapi function with this complete version:

def search_web_serpapi(query, max_results=5):
    """Search using SerpAPI (primary method) - Enhanced to capture URLs"""
    if not SERP_API_KEY:
        print("⚠️ SERP_API_KEY not found in environment")
        return [], []
    
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
        results = search.get_dict()  # This line was missing!

        if "error" in results:
            print(f"❌ SerpAPI Error: {results['error']}")
            return [], []

        texts = []
        links = []  # Add this line
        organic_results = results.get("organic_results", [])
        
        for i, r in enumerate(organic_results[:max_results]):
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            url = r.get("link", "")  # Add this line
            
            # Combine title and snippet for richer context
            combined_text = f"{title}. {snippet}" if title and snippet else (title or snippet)
            
            if combined_text:
                texts.append(combined_text)
                links.append(url)  # Add this line
                print(f"   Result {i+1}: {combined_text[:100]}...")
                
        print(f"✅ Found {len(texts)} results from SerpAPI")
        return texts, links  # Modified return statement
        
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return [], []
    

# 3. Modify search_web function to handle URLs
def search_web(query, max_results=5):
    print(f"\n🔍 Starting web search for: '{query}'")
    
    # Try SerpAPI first
    if SERP_API_KEY:
        results, links = search_web_serpapi(query, max_results)  # Modified
        if results:
            return results, links  # Modified return
    
    # Try DuckDuckGo as fallback
    print("🔄 Trying DuckDuckGo as fallback...")
    results, links = search_web_duckduckgo(query, max_results)  # Modified
    if results:
        return results, links  # Modified return
    
    print("❌ All search methods failed...")
    return [], []  # Modified return

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
# 4. Modify ingest_chunks function to store URLs with chunks
def ingest_chunks(chunks, links):
    """Ingest text chunks with their source URLs into the vector store"""
    if not chunks:
        print("⚠️ No chunks to ingest")
        return False
        
    try:
        print(f"💾 Ingesting {len(chunks)} chunks into vector store...")
        
        # Clear existing documents
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
        
        # Create metadata with URLs
        metadatas = []
        chunk_idx = 0
        for i, link in enumerate(links):
            # Assuming each text generated 1 chunk (adjust based on your chunking logic)
            metadatas.append({"source_url": link, "chunk_index": chunk_idx})
            chunk_idx += 1
        
        # Add to collection with metadata
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas  # Add this line
        )
        print(f"✅ Successfully ingested {len(chunks)} chunks with URLs")
        return True
        
    except Exception as e:
        print(f"❌ Error ingesting chunks: {e}")
        return False

# 5. Modify vector_search function to return URLs
def vector_search(query, k=3):
    """Perform semantic search to find relevant chunks with URLs"""
    try:
        print(f"🔎 Performing vector search (top-{k})...")
        cleaned_query = clean_text(query)
        query_emb = embed_model.encode([cleaned_query]).tolist()[0]

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "distances", "metadatas"]  # Add metadatas
        )
        
        docs = results["documents"][0]
        dists = results["distances"][0]
        metadatas = results["metadatas"][0]  # Add this line

        if not docs:
            print("❌ No matching documents found")
            return "", []

        print(f"\n📋 Top {len(docs)} matching chunks:")
        for i, (doc, dist) in enumerate(zip(docs, dists)):
            print(f"   Match {i+1} (similarity: {dist:.3f}): {doc[:150]}...")

        # Combine all relevant chunks
        context = "\n\n".join(docs)
        
        # Extract unique URLs
        source_urls = []
        for metadata in metadatas:
            url = metadata.get("source_url", "")
            if url and url not in source_urls:
                source_urls.append(url)
        
        print(f"✅ Retrieved {len(docs)} chunks for context")
        return context, source_urls  # Modified return
        
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        return "", []

# 7. Answer Generation
def generate_answer(query, context, source_urls):
    """Generate detailed, structured answer using LLM with retrieved context"""
    if not context or not context.strip():
        return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
        
    # Enhanced prompt for detailed responses
    prompt = f"""You are a knowledgeable AI assistant. Answer the user's question using the provided context in a detailed, well-structured manner.

Context Information:
{context}

User Question: {query}

Instructions for your response:
1. Provide a comprehensive answer using ONLY the information from the context above
2. Structure your response clearly with:
   - A brief introduction/overview
   - Main points organized with bullet points or numbered lists when appropriate
   - Specific details and examples from the context
   - A conclusion or summary if relevant
3. Use markdown formatting for better readability:
   - **Bold** for important terms and headings
   - *Italics* for emphasis
   - Bullet points (•) or numbered lists for key information
   - Line breaks for better organization
4. If the context doesn't fully answer the question, clearly state what information is available and what might be missing
5. Be specific and cite relevant details from the context
6. Maintain a professional, informative tone
7. Aim for 200-500 words depending on the complexity of the question

Your detailed response:"""
    
    try:
        print("✨ Generating detailed answer with LLM...")
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,  # Add some creativity while maintaining accuracy
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 1000  # Allow longer responses
            }
        }, timeout=OLLAMA_TIMEOUT * 2)  # Double timeout for longer responses
        
        if res.status_code == 200:
            answer = res.json()["response"].strip()
            
            # Post-process the answer for better formatting
            answer = format_response(answer)
            
            # Add sources section
            if source_urls:
                answer += "\n\n---\n\n**📚 Sources:**"
                for i, url in enumerate(source_urls[:5], 1):
                    # Try to extract domain name for cleaner display
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        answer += f"\n{i}. [{domain}]({url})"
                    except:
                        answer += f"\n{i}. {url}"
            
            print("✅ Detailed answer generated successfully")
            return answer
        else:
            return f"Error generating answer: HTTP {res.status_code}"
            
    except Exception as e:
        return f"Error generating answer: {e}"

def format_response(text):
    """Post-process the response for better formatting"""
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Convert simple lists to proper markdown
        if line.startswith('- ') or line.startswith('* '):
            formatted_lines.append(f"• {line[2:]}")
        elif line.startswith(tuple(f"{i}." for i in range(1, 10))):
            formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

# Enhanced query classification for better response types
def classify_query_detailed(query):
    """Enhanced query classification to determine response style"""
    cleaned = clean_text(query)
    prompt = f"""
Classify the user query to determine the best response approach. Reply with ONE label:

- "explanation" - User wants detailed explanation of concepts, processes, or topics
- "comparison" - User wants to compare multiple things  
- "step-by-step" - User wants instructions or procedures
- "factual" - User wants specific facts, numbers, or data points
- "analysis" - User wants analysis, pros/cons, or evaluation
- "summary" - User wants overview or summary of topic
- "troubleshooting" - User has a problem to solve
- "news" - User wants current events or recent news
- "definition" - User wants to understand what something means

Examples:
- "explain machine learning" → explanation
- "how to bake a cake" → step-by-step  
- "what happened in the news today" → news
- "compare iPhone vs Samsung" → comparison
- "what is blockchain" → definition
- "analyze pros and cons of remote work" → analysis

Query: "{query}"
Classification:"""
    
    try:
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=OLLAMA_TIMEOUT)
        
        if res.status_code == 200:
            classification = res.json()["response"].strip().lower()
            valid_types = ["explanation", "comparison", "step-by-step", "factual", 
                          "analysis", "summary", "troubleshooting", "news", "definition"]
            
            for query_type in valid_types:
                if query_type in classification:
                    print(f"✅ Query type: {query_type}")
                    return query_type
            
            return "explanation"  # Default fallback
        else:
            return "explanation"
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return "explanation"

def generate_structured_answer(query, context, source_urls, query_type="explanation"):
    """Generate answer based on query type for optimal structure"""
    if not context or not context.strip():
        return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
    
    # Different prompt templates based on query type
    prompts = {
        "explanation": f"""Provide a comprehensive explanation of the topic using the context below.

Context: {context}

Question: {query}

Structure your explanation as:
**Overview:** Brief introduction to the topic

**Key Concepts:**
• Main concept 1: Detailed explanation
• Main concept 2: Detailed explanation  
• Main concept 3: Detailed explanation

**Important Details:**
Include specific information, examples, and relevant details from the context.

**Summary:** 
Conclude with key takeaways or implications.

Your detailed explanation:""",

        "step-by-step": f"""Provide step-by-step instructions or process explanation using the context below.

Context: {context}

Question: {query}

Format your response as:
**Overview:** What this process accomplishes

**Steps:**
1. **Step 1:** Detailed description
2. **Step 2:** Detailed description  
3. **Step 3:** Detailed description
(Continue as needed)

**Important Notes:**
• Key considerations or warnings
• Tips for success
• Common mistakes to avoid

Your step-by-step guide:""",

        "comparison": f"""Compare the items mentioned in the question using the context below.

Context: {context}

Question: {query}

Structure your comparison as:
**Overview:** Brief introduction to what's being compared

**Key Differences:**
| Aspect | Option A | Option B |
|--------|----------|----------|
| Feature 1 | Details | Details |
| Feature 2 | Details | Details |

**Similarities:**
• Common feature 1
• Common feature 2

**Conclusion:** Which might be better and when

Your detailed comparison:""",

        "analysis": f"""Provide a thorough analysis using the context below.

Context: {context}

Question: {query}

Structure your analysis as:
**Background:** Context and current situation

**Key Factors:**
• **Factor 1:** Analysis and implications
• **Factor 2:** Analysis and implications
• **Factor 3:** Analysis and implications

**Pros and Cons:**
**Advantages:**
• Pro 1: Explanation
• Pro 2: Explanation

**Disadvantages:**
• Con 1: Explanation  
• Con 2: Explanation

**Conclusion:** Overall assessment and recommendations

Your detailed analysis:"""
    }
    
    # Use appropriate prompt or fallback to explanation
    selected_prompt = prompts.get(query_type, prompts["explanation"])
    
    try:
        print(f"✨ Generating {query_type} response...")
        res = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": selected_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 1500  # Allow even longer responses
            }
        }, timeout=OLLAMA_TIMEOUT * 3)  # Triple timeout for complex responses
        
        if res.status_code == 200:
            answer = res.json()["response"].strip()
            
            # Add sources section
            if source_urls:
                answer += "\n\n---\n\n**📚 Sources & References:**"
                for i, url in enumerate(source_urls[:5], 1):
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        answer += f"\n{i}. [{domain}]({url})"
                    except:
                        answer += f"\n{i}. {url}"
            
            print("✅ Structured answer generated successfully")
            return answer
        else:
            return f"Error generating answer: HTTP {res.status_code}"
            
    except Exception as e:
        return f"Error generating answer: {e}"# 8. Math Query Handler
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
    """Enhanced main function to process user queries with detailed responses"""
    print(f"\n{'='*60}")
    print(f"🚀 Processing query: '{user_input}'")
    print(f"{'='*60}")

    # Classify query type for optimal response structure
    query_type = classify_query_detailed(user_input)
    
    # Handle math queries directly
    if "math" in user_input.lower() or any(op in user_input for op in ['+', '-', '*', '/', '=', 'calculate', 'compute']):
        math_result = handle_math_query(user_input)
        if math_result:
            return math_result

    # Search the web for current information
    web_texts, web_links = search_web(user_input)
    
    if not web_texts:
        return "I couldn't find current information about your query. Please try rephrasing or ask about a different topic."

    # Process and chunk the web content
    print(f"\n📝 Processing {len(web_texts)} web results...")
    all_chunks = []
    chunk_links = []
    
    for i, (text, link) in enumerate(zip(web_texts, web_links)):
        if text.strip():
            cleaned = clean_text(text)
            if cleaned:
                chunks = chunk_text(text, max_words=150)  # Slightly larger chunks for more context
                all_chunks.extend(chunks)
                chunk_links.extend([link] * len(chunks))
                print(f"   Text {i+1}: Generated {len(chunks)} chunks")

    if not all_chunks:
        return "I found some web results but couldn't process them properly. Please try a different query."

    # Ingest into vector store with URLs
    if not ingest_chunks(all_chunks, chunk_links):
        return "There was an error processing the information. Please try again."

    # Retrieve more relevant context for detailed responses
    context, source_urls = vector_search(user_input, k=8)  # Get more chunks for detailed responses
    
    if not context:
        return "I couldn't find relevant information in the search results. Please try rephrasing your query."

    # Generate structured answer based on query type
    answer = generate_structured_answer(user_input, context, source_urls, query_type)
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