"""
Enhanced Dynamic RAG (Retrieval-Augmented Generation) System
===========================================================
This system intelligently determines when to search the web vs handle 
queries conversationally, preventing unnecessary web searches for greetings
and casual conversation.

Features:
- Smart query filtering (conversational vs informational)
- Real-time web search with SerpAPI (only when needed)
- Query classification for different response types
- Text chunking and vector embeddings
- Semantic search with ChromaDB 
- AI answer generation with Ollama
- Math query handling
- Source URL tracking
"""

# ================================
# IMPORTS AND SETUP
# ================================

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

# ================================
# CONFIGURATION
# ================================

# Get API keys and settings from environment
SERP_API_KEY = os.getenv("SERP_API_KEY")          # SerpAPI key for web search
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Ollama server URL
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")               # AI model name
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))          # Request timeout

# Display configuration status
print(f"🔧 Configuration loaded:")
print(f"   SERP_API_KEY: {'✅ Set' if SERP_API_KEY else '❌ Missing'}")
print(f"   OLLAMA_HOST: {OLLAMA_HOST}")
print(f"   OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"   OLLAMA_TIMEOUT: {OLLAMA_TIMEOUT}s")

# ================================
# GLOBAL AI COMPONENTS SETUP
# ================================

def setup_ai_components():
    """
    Initialize the AI components needed for the RAG system
    
    Returns:
        tuple: (embedding_model, vector_database_collection)
    """
    
    # Step 1: Load text embedding model
    # This converts text into numerical vectors for similarity search
    try:
        print("🧠 Loading embedding model...")
        # Try to load the advanced Nomic model first
        embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        print("✅ Nomic embedding model loaded successfully")
    except Exception as e:
        # If Nomic fails, use the standard model as fallback
        print(f"⚠️ Error loading Nomic model: {e}")
        print("🔄 Falling back to all-MiniLM-L6-v2...")
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Fallback embedding model loaded successfully")
        except Exception as e2:
            print(f"❌ Failed to load any embedding model: {e2}")
            exit(1)

    # Step 2: Initialize ChromaDB vector database
    # This stores and searches text embeddings efficiently
    try:
        print("🗄️ Initializing ChromaDB...")
        # Create persistent database that saves data to disk
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        # Create or get collection for storing our knowledge
        collection = chroma_client.get_or_create_collection("dynamic_knowledge")
        print("✅ ChromaDB initialized successfully")
        return embed_model, collection
    except Exception as e:
        print(f"❌ ChromaDB initialization error: {e}")
        exit(1)

# Initialize global AI components
embedding_model, vector_collection = setup_ai_components()

# ================================
# CONVERSATIONAL QUERY DETECTION
# ================================

def is_conversational_query(query):
    """
    Determine if a query is conversational/casual vs informational
    
    Args:
        query (str): User's input
        
    Returns:
        bool: True if conversational, False if needs web search
    """
    # Clean and normalize query
    cleaned_query = query.lower().strip()
    
    # Define conversational patterns
    conversational_patterns = [
        # Greetings
        r'^(hi|hello|hey|good morning|good afternoon|good evening)$',
        r'^(hi|hello|hey)\s*(there|!|\.|,)?$',
        
        # How are you variations
        r'^how\s+(are\s+you|r\s+u)(\?|!|\.)?$',
        r'^how\s+are\s+things(\?|!|\.)?$',
        r'^how\s+have\s+you\s+been(\?|!|\.)?$',
        
        # Basic responses
        r'^(good|fine|okay|ok|great|awesome|nice)(\s+thanks?)?(\?|!|\.)?$',
        r'^(thank\s+you|thanks?)(\s+so\s+much)?(\?|!|\.)?$',
        r'^you\'?re\s+welcome(\?|!|\.)?$',
        
        # Simple questions about the AI
        r'^what\s+(are\s+you|r\s+u)(\?|!|\.)?$',
        r'^who\s+(are\s+you|r\s+u)(\?|!|\.)?$',
        r'^what\s+is\s+your\s+name(\?|!|\.)?$',
        r'^what\s+can\s+you\s+do(\?|!|\.)?$',
        
        # Casual farewells
        r'^(bye|goodbye|see\s+you|talk\s+to\s+you\s+later|ttyl)(\?|!|\.)?$',
        
        # Simple acknowledgments
        r'^(yes|yeah|yep|no|nope|maybe|i\s+see|ok|okay)(\?|!|\.)?$',
        
        # Weather small talk (when very simple)
        r'^nice\s+day(\?|!|\.)?$',
        r'^how\s+is\s+the\s+weather(\?|!|\.)?$',
    ]
    
    # Check against patterns
    for pattern in conversational_patterns:
        if re.match(pattern, cleaned_query):
            return True
    
    # Additional heuristics
    # Very short queries (1-3 words) that don't look informational
    words = cleaned_query.split()
    if len(words) <= 3:
        non_informational_words = {
            'hi', 'hello', 'hey', 'thanks', 'thank', 'you', 'good', 'fine', 
            'ok', 'okay', 'yes', 'no', 'bye', 'goodbye', 'nice', 'great',
            'awesome', 'cool', 'wow', 'amazing', 'interesting'
        }
        if all(word in non_informational_words for word in words):
            return True
    
    return False

def handle_conversational_query(query):
    """
    Generate responses for conversational queries without web search
    
    Args:
        query (str): Conversational query
        
    Returns:
        str: Appropriate conversational response
    """
    cleaned_query = query.lower().strip()
    
    # Define response patterns
    if re.match(r'^(hi|hello|hey)', cleaned_query):
        return "Hello! I'm here to help answer your questions. What would you like to know about?"
    
    elif re.match(r'^how\s+(are\s+you|r\s+u)', cleaned_query):
        return "I'm doing well, thank you for asking! I'm ready to help you find information or answer questions. What can I assist you with today?"
    
    elif re.match(r'^(thank\s+you|thanks)', cleaned_query):
        return "You're very welcome! Is there anything else I can help you with?"
    
    elif re.match(r'^what\s+(are\s+you|r\s+u)', cleaned_query):
        return "I'm an AI assistant powered by a RAG (Retrieval-Augmented Generation) system. I can search the web for current information and provide detailed answers to your questions!"
    
    elif re.match(r'^who\s+(are\s+you|r\s+u)', cleaned_query):
        return "I'm an AI assistant that can help you find information by searching the web and providing comprehensive answers. Feel free to ask me about any topic!"
    
    elif re.match(r'^what\s+can\s+you\s+do', cleaned_query):
        return """I can help you with many things:
• Answer questions by searching current web information
• Explain complex topics with detailed breakdowns
• Provide step-by-step instructions
• Compare different options or products
• Analyze situations and provide insights
• Handle math calculations
• And much more! Just ask me anything you're curious about."""
    
    elif re.match(r'^(bye|goodbye)', cleaned_query):
        return "Goodbye! Feel free to come back anytime if you have more questions. Have a great day!"
    
    elif re.match(r'^(good|fine|okay|ok|great|awesome|nice)', cleaned_query):
        return "That's great to hear! What would you like to explore or learn about today?"
    
    else:
        # Generic conversational response
        return "I'm here and ready to help! What would you like to know or discuss?"

# ================================
# TEXT PROCESSING UTILITIES
# ================================

def clean_text(text):
    """
    Clean and normalize text for better processing
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Handle empty or None text
    if not text:
        return ""
    
    # Convert to lowercase and remove extra whitespace
    text = text.strip().lower()
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation for better matching
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def chunk_text(text, max_words=100):
    """
    Split long text into smaller chunks for better processing
    
    Args:
        text (str): Text to chunk
        max_words (int): Maximum words per chunk
        
    Returns:
        list: List of text chunks
    """
    # Handle empty text
    if not text or not text.strip():
        return []
    
    # Split text into words
    words = text.split()
    chunks = []
    
    # Create chunks of specified size
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

# ================================
# QUERY CLASSIFICATION SYSTEM
# ================================

def classify_query_type(query):
    """
    Classify user query to determine the best processing approach
    
    Args:
        query (str): User's question/query
        
    Returns:
        str: Classification label (real-time, static, code, math, etc.)
    """
    # Clean the query for analysis
    cleaned_query = clean_text(query)
    
    # Create classification prompt for AI
    classification_prompt = f"""
Classify the user query below. Reply only with one label from:
real-time, static, code, math, search, tool:calendar, tool:news

Examples:
- "what is python" -> static
- "how to code in python" -> code  
- "what is 2+2" -> math
- "current weather" -> real-time
- "news today" -> tool:news

Query: "{cleaned_query}"
Classification:"""
    
    try:
        print("🤖 Classifying query type...")
        
        # Send classification request to Ollama
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": classification_prompt,
            "stream": False
        }, timeout=OLLAMA_TIMEOUT)
        
        # Process the response
        if response.status_code == 200:
            classification = response.json()["response"].strip().lower()
            
            # Extract the classification label
            valid_labels = ["real-time", "static", "code", "math", "search", "tool:calendar", "tool:news"]
            for label in valid_labels:
                if label in classification:
                    print(f"✅ Query classified as: {label}")
                    return label
            
            # Default fallback
            print("⚠️ Could not determine classification, defaulting to 'static'")
            return "static"
        else:
            print(f"❌ LLM API error: {response.status_code}")
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

def classify_response_type(query):
    """
    Classify query to determine the best response structure
    
    Args:
        query (str): User's question
        
    Returns:
        str: Response type (explanation, comparison, step-by-step, etc.)
    """
    cleaned_query = clean_text(query)
    
    # Create detailed classification prompt
    response_type_prompt = f"""
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
        # Send request to AI
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": response_type_prompt,
            "stream": False
        }, timeout=OLLAMA_TIMEOUT)
        
        if response.status_code == 200:
            classification = response.json()["response"].strip().lower()
            
            # Valid response types
            valid_types = ["explanation", "comparison", "step-by-step", "factual", 
                          "analysis", "summary", "troubleshooting", "news", "definition"]
            
            # Find matching type
            for response_type in valid_types:
                if response_type in classification:
                    print(f"✅ Response type: {response_type}")
                    return response_type
            
            # Default fallback
            return "explanation"
        else:
            return "explanation"
            
    except Exception as e:
        print(f"❌ Response type classification error: {e}")
        return "explanation"

# ================================
# WEB SEARCH SYSTEM
# ================================

def search_with_serpapi(query, max_results=5):
    """
    Search the web using SerpAPI (Google Search)
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        
    Returns:
        tuple: (list of texts, list of URLs)
    """
    # Check if API key is available
    if not SERP_API_KEY:
        print("⚠️ SERP_API_KEY not found in environment")
        return [], []
    
    try:
        print(f"🔍 Searching with SerpAPI: '{query}'")
        
        # Set up search parameters
        search_params = {
            "q": query,                    # Search query
            "api_key": SERP_API_KEY,      # API key
            "engine": "google",            # Use Google search
            "num": max_results,            # Number of results
            "gl": "us",                    # Geographic location
            "hl": "en"                     # Language
        }

        # Perform the search
        search = GoogleSearch(search_params)
        results = search.get_dict()

        # Check for errors
        if "error" in results:
            print(f"❌ SerpAPI Error: {results['error']}")
            return [], []

        # Extract information from results
        texts = []
        urls = []
        organic_results = results.get("organic_results", [])
        
        # Process each search result
        for i, result in enumerate(organic_results[:max_results]):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            url = result.get("link", "")
            
            # Combine title and snippet for richer context
            combined_text = f"{title}. {snippet}" if title and snippet else (title or snippet)
            
            # Add to our collections if we have content
            if combined_text:
                texts.append(combined_text)
                urls.append(url)
                print(f"   Result {i+1}: {combined_text[:100]}...")
                
        print(f"✅ Found {len(texts)} results from SerpAPI")
        return texts, urls
        
    except Exception as e:
        print(f"❌ SerpAPI error: {e}")
        return [], []

def search_web(query, max_results=5):
    """
    Main web search function with fallback options
    
    Args:
        query (str): Search query
        max_results (int): Maximum results to return
        
    Returns:
        tuple: (list of texts, list of URLs)
    """
    print(f"\n🔍 Starting web search for: '{query}'")
    
    # Try SerpAPI first (primary method)
    if SERP_API_KEY:
        results, links = search_with_serpapi(query, max_results)
        if results:
            return results, links
    
    # If SerpAPI fails, we could add other search methods here
    # For now, just return empty results
    print("❌ All search methods failed...")
    return [], []

# ================================
# VECTOR STORAGE SYSTEM
# ================================

def store_text_chunks(chunks, source_urls):
    """
    Store text chunks in the vector database with their source URLs
    
    Args:
        chunks (list): List of text chunks to store
        source_urls (list): List of URLs corresponding to each chunk
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if we have chunks to store
    if not chunks:
        print("⚠️ No chunks to store")
        return False
        
    try:
        print(f"💾 Storing {len(chunks)} chunks in vector database...")
        
        # Step 1: Clear existing documents to avoid duplicates
        try:
            existing_docs = vector_collection.get()
            if existing_docs['ids']:
                vector_collection.delete(ids=existing_docs['ids'])
                print(f"🗑️ Cleared {len(existing_docs['ids'])} existing documents")
        except Exception as e:
            print(f"⚠️ Could not clear existing documents: {e}")
        
        # Step 2: Generate embeddings (convert text to numbers)
        print("🔢 Generating embeddings...")
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Step 3: Prepare data for storage
        # Create unique IDs for each chunk
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Create metadata with source URLs
        metadata_list = []
        for i, url in enumerate(source_urls):
            metadata_list.append({
                "source_url": url,
                "chunk_index": i
            })
        
        # Step 4: Store everything in the database
        vector_collection.add(
            documents=chunks,           # The actual text
            embeddings=embeddings,      # Numerical representations
            ids=chunk_ids,             # Unique identifiers
            metadatas=metadata_list    # Additional information
        )
        
        print(f"✅ Successfully stored {len(chunks)} chunks with URLs")
        return True
        
    except Exception as e:
        print(f"❌ Error storing chunks: {e}")
        return False

def search_similar_chunks(query, top_k=3):
    """
    Find the most similar chunks to the query using vector search
    
    Args:
        query (str): User's question
        top_k (int): Number of most similar chunks to return
        
    Returns:
        tuple: (combined_context, list_of_source_urls)
    """
    try:
        print(f"🔎 Performing semantic search (top-{top_k})...")
        
        # Step 1: Convert query to embedding
        cleaned_query = clean_text(query)
        query_embedding = embedding_model.encode([cleaned_query]).tolist()[0]

        # Step 2: Search for similar chunks
        search_results = vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )
        
        # Step 3: Extract results
        documents = search_results["documents"][0]
        distances = search_results["distances"][0]
        metadatas = search_results["metadatas"][0]

        # Step 4: Check if we found anything
        if not documents:
            print("❌ No matching documents found")
            return "", []

        # Step 5: Display what we found
        print(f"\n📋 Top {len(documents)} matching chunks:")
        for i, (doc, dist) in enumerate(zip(documents, distances)):
            print(f"   Match {i+1} (similarity: {dist:.3f}): {doc[:150]}...")

        # Step 6: Combine all relevant chunks into context
        combined_context = "\n\n".join(documents)
        
        # Step 7: Extract unique source URLs
        source_urls = []
        for metadata in metadatas:
            url = metadata.get("source_url", "")
            if url and url not in source_urls:
                source_urls.append(url)
        
        print(f"✅ Retrieved {len(documents)} chunks for context")
        return combined_context, source_urls
        
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        return "", []

# ================================
# MATH QUERY HANDLER
# ================================

def handle_math_calculation(query):
    """
    Handle direct mathematical calculations
    
    Args:
        query (str): Math query from user
        
    Returns:
        str or None: Result if calculation successful, None otherwise
    """
    try:
        print("🧮 Attempting direct math calculation...")
        
        # Step 1: Clean the query to extract mathematical expression
        math_expression = query.lower()
        
        # Remove common words
        math_expression = math_expression.replace("what is", "")
        math_expression = math_expression.replace("calculate", "")
        math_expression = math_expression.replace("compute", "")
        math_expression = math_expression.replace("solve", "")
        
        # Replace common symbols
        math_expression = math_expression.replace("×", "*")
        math_expression = math_expression.replace("÷", "/")
        math_expression = math_expression.replace("plus", "+")
        math_expression = math_expression.replace("minus", "-")
        math_expression = math_expression.strip()
        
        # Step 2: Safety check - only allow basic math operations
        allowed_characters = set("0123456789+-*/.()\s")
        if all(c in allowed_characters for c in math_expression) and math_expression:
            # Step 3: Calculate the result
            result = eval(math_expression)
            print(f"✅ Direct calculation: {math_expression} = {result}")
            return f"The answer is: {result}"
        else:
            print("🔍 Complex math expression, will search for context...")
            return None
            
    except Exception as e:
        print("⚠️ Could not compute directly, will search for context...")
        return None

# ================================
# ANSWER GENERATION SYSTEM
# ================================

def generate_structured_answer(query, context, source_urls, response_type="explanation"):
    """
    Generate a structured answer based on the query type and context
    
    Args:
        query (str): User's question
        context (str): Retrieved information
        source_urls (list): List of source URLs
        response_type (str): Type of response to generate
        
    Returns:
        str: Generated answer
    """
    # Handle empty context
    if not context or not context.strip():
        return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
    
    # Different prompt templates based on response type
    prompt_templates = {
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
    
    # Select appropriate prompt template
    selected_prompt = prompt_templates.get(response_type, prompt_templates["explanation"])
    
    try:
        print(f"✨ Generating {response_type} response...")
        
        # Send request to AI model
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": selected_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,      # Controls creativity
                "top_k": 40,            # Limits vocabulary choices
                "top_p": 0.9,           # Controls response diversity
                "num_predict": 1500     # Maximum response length
            }
        }, timeout=OLLAMA_TIMEOUT * 3)  # Extra time for complex responses
        
        if response.status_code == 200:
            answer = response.json()["response"].strip()
            
            # Add sources section to the answer
            if source_urls:
                answer += "\n\n---\n\n**📚 Sources & References:**"
                for i, url in enumerate(source_urls[:5], 1):
                    try:
                        # Extract domain name for cleaner display
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        answer += f"\n{i}. [{domain}]({url})"
                    except:
                        answer += f"\n{i}. {url}"
            
            print("✅ Structured answer generated successfully")
            return answer
        else:
            return f"Error generating answer: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error generating answer: {e}"

# ================================
# MAIN PROCESSING PIPELINE
# ================================

def process_user_query(user_input):
    """
    Main function that processes user queries through the complete RAG pipeline
    
    Args:
        user_input (str): User's question or query
        
    Returns:
        str: Generated answer
    """
    print(f"\n{'='*60}")
    print(f"🚀 Processing query: '{user_input}'")
    print(f"{'='*60}")

    # Step 0: Check if this is a conversational query
    if is_conversational_query(user_input):
        print("💬 Detected conversational query - responding without web search")
        return handle_conversational_query(user_input)

    # Step 1: Classify the query to determine response type
    response_type = classify_response_type(user_input)
    
    # Step 2: Check if this is a math query and handle it directly
    if "math" in user_input.lower() or any(op in user_input for op in ['+', '-', '*', '/', '=', 'calculate', 'compute']):
        math_result = handle_math_calculation(user_input)
        if math_result:
            return math_result

    # Step 3: Search the web for current information
    web_texts, web_urls = search_web(user_input)
    
    # Check if we found any web results
    if not web_texts:
        return "I couldn't find current information about your query. Please try rephrasing or ask about a different topic."

    # Step 4: Process and chunk the web content
    print(f"\n📝 Processing {len(web_texts)} web results...")
    all_text_chunks = []
    chunk_source_urls = []
    
    # Process each web result
    for i, (text, url) in enumerate(zip(web_texts, web_urls)):
        if text.strip():
            # Clean the text
            cleaned_text = clean_text(text)
            
            if cleaned_text:
                text_chunks = chunk_text(text, max_words=150)
                all_text_chunks.extend(text_chunks)
                
                # Associate each chunk with its source URL
                chunk_source_urls.extend([url] * len(text_chunks))
                print(f"   Text {i+1}: Generated {len(text_chunks)} chunks")

    # Check if we have chunks to work with
    if not all_text_chunks:
        return "I found some web results but couldn't process them properly. Please try a different query."

    # Step 5: Store chunks in vector database
    if not store_text_chunks(all_text_chunks, chunk_source_urls):
        return "There was an error processing the information. Please try again."

    # Step 6: Retrieve most relevant chunks for the query
    relevant_context, source_urls = search_similar_chunks(user_input, top_k=8)
    
    # Check if we found relevant information
    if not relevant_context:
        return "I couldn't find relevant information in the search results. Please try rephrasing your query."

    # Step 7: Generate the final structured answer
    final_answer = generate_structured_answer(user_input, relevant_context, source_urls, response_type)
    return final_answer

# ================================
# MAIN APPLICATION LOOP
# ================================

def main():
    """
    Main interactive loop for the RAG system
    """
    # Display welcome message
    print("🤖 Enhanced Dynamic RAG System Ready!")
    print("💡 This system intelligently determines when to search vs respond conversationally")
    print("🔄 Type 'quit' to exit\n")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\n📝 Enter your query: ").strip()
            
            # Check for empty input
            if not user_input:
                print("⚠️ Please enter a valid query")
                continue
                
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            # Process the user's query
            answer = process_user_query(user_input)
            
            # Display the final answer
            print(f"\n{'='*60}")
            print("🎯 FINAL ANSWER:")
            print(f"{'='*60}")
            print(answer)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            # Handle unexpected errors
            print(f"❌ Unexpected error: {e}")
            print("Please try again with a different query.")

# ================================
# PROGRAM ENTRY POINT
# ================================

if __name__ == "__main__":
    main()

# ================================
# ADDITIONAL IMPROVEMENTS YOU CAN ADD
# ================================

def update_conversational_patterns():
    """
    Additional conversational patterns you can add to is_conversational_query function
    """
    additional_patterns = [
        # More casual expressions
        r'^(sup|wassup|what\'?s up)(\?|!|\.)?$',
        r'^how\s+is\s+it\s+going(\?|!|\.)?$',
        r'^(cool|awesome|amazing|interesting)(\s+stuff)?(\?|!|\.)?$',
        
        # Simple acknowledgments
        r'^(i\s+understand|got\s+it|makes\s+sense)(\?|!|\.)?$',
        r'^(that\'?s\s+helpful|very\s+useful)(\?|!|\.)?$',
        
        # Simple requests without needing web search
        r'^(help\s+me|can\s+you\s+help)(\?|!|\.)?$',
        r'^what\s+should\s+i\s+ask(\?|!|\.)?$',
    ]
    return additional_patterns

def enhanced_math_detection(query):
    """
    Enhanced math detection patterns
    """
    math_indicators = [
        r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Basic operations
        r'calculate|compute|solve|find',  # Math keywords
        r'percentage|percent|\%',  # Percentage calculations
        r'square\s+root|sqrt',  # Square root
        r'power|exponent|\^',  # Powers
        r'average|mean|median',  # Statistics
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, query.lower()):
            return True
    return False

def add_query_preprocessing(query):
    """
    Additional query preprocessing before web search
    """
    # Remove common filler words that don't help search
    filler_words = ['please', 'can you', 'could you', 'i want to know', 'tell me about']
    
    processed_query = query.lower()
    for filler in filler_words:
        processed_query = processed_query.replace(filler, '')
    
    # Clean up extra spaces
    processed_query = ' '.join(processed_query.split())
    
    return processed_query if processed_query else query

# ================================
# PERFORMANCE OPTIMIZATION SUGGESTIONS
# ================================

def optimize_chunk_storage():
    """
    Optimization: Only store new chunks if query context changes significantly
    """
    # You can implement a hash-based system to avoid re-processing identical content
    import hashlib
    
    def get_content_hash(chunks):
        content_string = "".join(chunks)
        return hashlib.md5(content_string.encode()).hexdigest()
    
    # Store hash with chunks and compare before re-processing
    pass

def implement_caching():
    """
    Optimization: Cache recent query results to avoid repeated web searches
    """
    import time
    from functools import lru_cache
    
    # Simple in-memory cache for recent queries
    query_cache = {}
    CACHE_DURATION = 300  # 5 minutes
    
    def get_cached_result(query):
        if query in query_cache:
            result, timestamp = query_cache[query]
            if time.time() - timestamp < CACHE_DURATION:
                return result
        return None
    
    def cache_result(query, result):
        query_cache[query] = (result, time.time())
    
    return get_cached_result, cache_result