# Enhanced Dynamic RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that intelligently determines when to search the web versus handle queries conversationally, preventing unnecessary web searches for greetings and casual conversation.

## 🌟 Features

- **Smart Query Classification**: Automatically distinguishes between conversational queries and informational requests
- **Real-time Web Search**: Uses SerpAPI for current information retrieval
- **Vector-based Semantic Search**: ChromaDB with sentence transformers for intelligent content matching
- **Multi-modal Response Generation**: Different response structures based on query type
- **Math Query Handling**: Direct calculation capabilities for mathematical expressions
- **Source Attribution**: Automatic citation of web sources
- **Conversational AI**: Natural dialogue handling without unnecessary web searches

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT PROCESSING                        │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONVERSATIONAL DETECTION                           │
│  • Pattern matching for greetings, casual chat                 │
│  • Returns direct response if conversational                    │
└─────────────────┬───────────────────────────────────────────────┘
                  │ (If NOT conversational)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                QUERY CLASSIFICATION                             │
│  • Classify into: real-time, static, code, math, etc.          │
│  • Determine response type: explanation, comparison, etc.       │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MATH DETECTION                                 │
│  • Direct calculation for simple math expressions              │
│  • Uses eval() for basic arithmetic                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │ (If NOT math)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   WEB SEARCH                                    │
│  • SerpAPI (Google Search) integration                         │
│  • Retrieves top-k results with titles & snippets              │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│               TEXT PROCESSING                                   │
│  • Clean and normalize retrieved text                          │
│  • Chunk text into manageable pieces (100-150 words)           │
│  • Maintain source URL associations                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              VECTOR STORAGE                                     │
│  • Generate embeddings using SentenceTransformer               │
│  • Store in ChromaDB with metadata                             │
│  • Clear previous documents to avoid duplicates                │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│             SEMANTIC SEARCH                                     │
│  • Convert user query to embedding                             │
│  • Find top-k most similar chunks                              │
│  • Retrieve relevant context + source URLs                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│          STRUCTURED ANSWER GENERATION                          │
│  • Use Ollama LLM with context                                 │
│  • Apply response-type specific templates                      │
│  • Include source citations                                     │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FINAL RESPONSE                                  │
│  • Formatted answer with sources                               │
│  • Return to user                                              │
└─────────────────────────────────────────────────────────────────┘


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama server running locally
- SerpAPI account and API key

### Installation

1. **Clone or download the system files**

2. **Install required dependencies:**
```bash
pip install sentence-transformers chromadb serpapi python-dotenv requests
```

3. **Set up Ollama:**
```bash
# Install Ollama (visit https://ollama.ai for instructions)
# Pull the required model
ollama pull llama3
```

4. **Create environment file:**
Create a `.env` file in the project directory:
```env
SERP_API_KEY=your_serpapi_key_here
OLLAMA_HOST=your_ollama_host_url
OLLAMA_MODEL=your_preferred_model
OLLAMA_TIMEOUT=your_timeout_value
```

5. **Run the system:**
```bash
python rag_system.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SERP_API_KEY` | SerpAPI key for web search | None | Yes |
| `OLLAMA_HOST` | Ollama server URL | localhost | No |
| `OLLAMA_MODEL` | AI model name | llama3 | No |
| `OLLAMA_TIMEOUT` | Request timeout in seconds | 30 | No |

### Getting SerpAPI Key

1. Visit [SerpAPI](https://serpapi.com/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Add it to your `.env` file

## 💬 Usage Examples

### Conversational Queries (No Web Search)
```
User: Hi there!
System: Hello! I'm here to help answer your questions. What would you like to know about?

User: How are you?
System: I'm doing well, thank you for asking! I'm ready to help you find information...

User: What can you do?
System: I can help you with many things:
• Answer questions by searching current web information
• Explain complex topics with detailed breakdowns
• Provide step-by-step instructions...
```

### Informational Queries (Web Search + AI Generation)
```
User: What are the latest developments in artificial intelligence?
System: [Searches web, processes results, generates structured response]

User: How do I bake a chocolate cake?
System: [Provides step-by-step instructions based on current web information]

User: Compare iPhone vs Samsung Galaxy
System: [Creates detailed comparison table with pros/cons]
```

### Math Queries (Direct Calculation)
```
User: What is 25 * 47?
System: The answer is: 1175

User: Calculate 15% of 847,293
System: [Performs calculation and provides result]
```

## 🧠 Query Classification System

The system automatically classifies queries into different types:

### Query Types
- **Conversational**: Greetings, casual chat, basic AI questions
- **Informational**: Requests for current information requiring web search
- **Mathematical**: Calculations and mathematical expressions
- **Code**: Programming-related queries
- **Real-time**: Current events, weather, news

### Response Types
- **Explanation**: Detailed concept explanations
- **Step-by-step**: Instructions and procedures
- **Comparison**: Side-by-side comparisons
- **Analysis**: Pros/cons and evaluations
- **Factual**: Specific data points and facts
- **Summary**: Topic overviews

## 🔍 Technical Components

### 1. Text Processing
- **Cleaning**: Normalization and preprocessing
- **Chunking**: Breaking long texts into manageable pieces
- **Embedding**: Converting text to numerical vectors

### 2. Vector Database (ChromaDB)
- **Storage**: Persistent vector storage
- **Search**: Semantic similarity search
- **Metadata**: Source URL tracking

### 3. Web Search (SerpAPI)
- **Real-time**: Current information retrieval
- **Multiple Sources**: Diverse web content
- **URL Tracking**: Source attribution

### 4. AI Generation (Ollama)
- **Local Processing**: Privacy-focused AI
- **Structured Responses**: Template-based generation
- **Customizable**: Multiple model support

## 📁 Project Structure

```
rag_system/
├── rag_system.py          # Main application file
├── .env                   # Environment variables
├── chroma_store/          # Vector database storage
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 🛠️ Customization

### Adding New Conversational Patterns
```python
def is_conversational_query(query):
    # Add your patterns to the conversational_patterns list
    conversational_patterns = [
        r'^your_custom_pattern$',
        # ... existing patterns
    ]
```

### Modifying Response Templates
```python
def generate_structured_answer():
    prompt_templates = {
        "your_custom_type": f"""Your custom prompt template...""",
        # ... existing templates
    }
```

### Adjusting Search Parameters
```python
def search_with_serpapi(query, max_results=5):
    search_params = {
        "num": max_results,    # Adjust number of results
        "gl": "us",           # Change geographic location
        "hl": "en"            # Change language
    }
```

## ⚡ Performance Optimization

### Recommended Settings
- **Chunk Size**: 100-150 words for optimal context
- **Vector Search**: Top 3-8 results for best relevance
- **Timeout**: Configure appropriate timeout values
- **Cache Duration**: 5-10 minutes for repeated queries

### Scaling Considerations
- Use batch processing for multiple queries
- Implement query caching for frequently asked questions
- Consider distributed vector storage for large datasets
- Monitor API rate limits

## 🔒 Privacy & Security

- **Local AI Processing**: Ollama runs locally for privacy
- **API Key Security**: Store keys in environment variables
- **Data Persistence**: ChromaDB stores data locally
- **No External Dependencies**: Minimal third-party services

## 🐛 Troubleshooting

### Common Issues

1. **"Cannot connect to Ollama"**
   - Ensure Ollama is installed and running
   - Check if the model is pulled: `ollama pull [model_name]`
   - Verify OLLAMA_HOST in .env file

2. **"SERP_API_KEY not found"**
   - Check .env file exists and has correct key
   - Verify API key is valid on SerpAPI dashboard

3. **"No matching documents found"**
   - Query might be too specific
   - Try rephrasing with more common terms
   - Check if web search returned results

4. **Slow Response Times**
   - Increase OLLAMA_TIMEOUT in .env
   - Reduce chunk size or top_k parameters
   - Check internet connection for web searches

### Debug Mode
Enable detailed logging by adding print statements or using Python's logging module:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Future Enhancements

### Planned Features
- [ ] Multi-language support
- [ ] Document upload and processing
- [ ] Advanced math computation
- [ ] Image and multimedia handling
- [ ] API endpoint creation
- [ ] Web interface development

### Performance Improvements
- [ ] Query result caching
- [ ] Batch processing capabilities
- [ ] Distributed vector storage
- [ ] Advanced conversation memory

## 📝 License

This project is open source. Please ensure compliance with all third-party service terms of use (SerpAPI, Ollama, etc.).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review environment configuration
- Ensure all dependencies are installed
- Verify API keys and services are working

## 🙏 Acknowledgments

- **Ollama**: Local AI model serving
- **SerpAPI**: Web search capabilities
- **ChromaDB**: Vector database storage
- **Sentence Transformers**: Text embeddings
- **Python Community**: Various supporting libraries

---

**Ready to explore knowledge with intelligent search! 🚀**