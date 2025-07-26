# Multilingual RAG System (Bengali-English)

A comprehensive Retrieval-Augmented Generation (RAG) system that processes Bengali and English queries, retrieves relevant information from PDF documents, and generates contextual answers using advanced NLP techniques.

## 🎯 Features

- **Multilingual Support**: Process queries in both Bengali and English
- **PDF Document Processing**: Extract and chunk text from PDF documents
- **Vector Database**: Store document embeddings for efficient retrieval
- **Conversation Memory**: Maintain short-term and long-term conversation history
- **REST API**: Easy integration with web applications
- **Comprehensive Evaluation**: Built-in performance metrics and testing
- **Interactive CLI**: Command-line interface for easy usage

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd multilingual-rag-system

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows
# OR
source venv/bin/activate      # On Linux/Mac

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# API Keys (Get these from OpenAI and Google)
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Customize settings
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=gpt-3.5-turbo
```

### 3. Add Your Documents

Place your PDF files in the `data/pdfs/` directory:

```bash
# Create directory structure
mkdir -p data/pdfs data/vector_db

# Add your PDF files
cp your_documents.pdf data/pdfs/
```

### 4. Setup Knowledge Base

```bash
# Setup knowledge base from all PDFs in data/pdfs/
python main.py setup

# Or specify specific files
python main.py setup --pdf-files data/pdfs/document1.pdf data/pdfs/document2.pdf
```

### 5. Start Using the System

#### Interactive Mode
```bash
python main.py query
```

#### API Server
```bash
python main.py api
# Access API docs at http://localhost:8000/docs
```

#### Run Tests
```bash
python main.py test
```

## 📖 Usage Guide

### Command Line Interface

The system provides several commands for different use cases:

#### 1. Setup Knowledge Base
```bash
# Setup from all PDFs in data/pdfs/
python main.py setup

# Setup from specific files
python main.py setup --pdf-files path/to/file1.pdf path/to/file2.pdf
```

#### 2. Interactive Query Session
```bash
python main.py query
```
This starts an interactive session where you can:
- Ask questions in Bengali or English
- Type `stats` to see system statistics
- Type `clear` to clear conversation history
- Type `quit` to exit

#### 3. Run Test Queries
```bash
python main.py test
```
Runs predefined test cases to verify system functionality.

#### 4. Start API Server
```bash
python main.py api
```
Starts the REST API server on `http://localhost:8000`

#### 5. Check System Status
```bash
python main.py status
```
Shows current system configuration and statistics.

#### 6. Run Evaluation
```bash
# Run evaluation with default test cases
python main.py evaluate

# Save evaluation report to file
python main.py evaluate --output evaluation_report.json
```

### REST API Usage

Once the API server is running, you can use these endpoints:

#### Query Endpoint
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "include_context": true,
    "top_k": 5
  }'
```

#### Upload PDF
```bash
curl -X POST "http://localhost:8000/upload-pdf" \
  -F "file=@your_document.pdf"
```

#### Get Statistics
```bash
curl "http://localhost:8000/stats"
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with these settings:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

# Optional: Customize Models
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_MODEL=gpt-3.5-turbo

# Optional: Database Settings
VECTOR_DB_PATH=data/vector_db
PDF_PATH=data/pdfs

# Optional: Chunking Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Optional: Retrieval Settings
TOP_K_CHUNKS=5
SIMILARITY_THRESHOLD=0.7

# Optional: API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Memory Settings
MAX_CONVERSATION_HISTORY=10
```

### Configuration File

You can also modify `config.py` to change default settings:

```python
# Example: Change chunk size
CHUNK_SIZE = 300  # Smaller chunks for more precise retrieval

# Example: Change similarity threshold
SIMILARITY_THRESHOLD = 0.8  # Higher threshold for more relevant results
```

## 📊 Sample Queries

### Bengali Queries
```
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
```

### English Queries
```
What is the main theme of the story?
Who is the protagonist in this narrative?
What are the social issues discussed?
```

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/
```

### Run Integration Tests
```bash
python main.py test
```

### Run Evaluation
```bash
python main.py evaluate --output results.json
```

### Demo Script
```bash
python demo.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, make sure you're in the virtual environment
source venv/Scripts/activate  # Windows
# OR
source venv/bin/activate      # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. API Key Issues
```bash
# Make sure your .env file exists and has valid API keys
cat .env
# Should show your API keys

# Test API connection
python -c "import openai; openai.api_key='your_key'; print('Valid key')"
```

#### 3. PDF Processing Issues
```bash
# Check if PDF files are readable
python -c "import PyPDF2; print('PyPDF2 works')"

# Try different PDF files if one fails
python main.py setup --pdf-files different_file.pdf
```

#### 4. Memory Issues
```bash
# Reduce chunk size for large documents
# Edit config.py: CHUNK_SIZE = 300

# Clear vector database and restart
rm -rf data/vector_db/*
python main.py setup
```

#### 5. Performance Issues
```bash
# Use smaller embedding model
# Edit config.py: EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2"

# Reduce top_k for faster retrieval
# Edit config.py: TOP_K_CHUNKS = 3
```

### Getting Help

1. **Check the logs**: Look for error messages in the terminal
2. **Verify setup**: Run `python main.py status` to check system health
3. **Test components**: Use `python main.py test` to verify functionality
4. **Check API docs**: Visit `http://localhost:8000/docs` when API is running

## 📁 Project Structure

```
multilingual-rag-system/
├── src/                    # Core source code
│   ├── pdf_processor.py    # PDF text extraction
│   ├── text_chunker.py     # Document chunking
│   ├── vector_database.py  # Vector storage & search
│   ├── conversation_memory.py # Memory management
│   ├── multilingual_rag.py # Main RAG system
│   ├── api.py             # REST API
│   └── evaluation.py      # Evaluation metrics
├── data/                  # Data storage
│   ├── pdfs/             # Source PDF documents
│   ├── vector_db/        # Vector database
│   └── conversations/    # Conversation history
├── tests/                # Test files
├── config.py             # Configuration
├── main.py              # CLI interface
├── demo.py              # Demo script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔄 Development Workflow

### 1. Setup Development Environment
```bash
# Clone and setup
git clone <repo-url>
cd multilingual-rag-system
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

# Install development dependencies
pip install black flake8 pytest
```

### 2. Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Run tests
pytest tests/
```

### 3. Testing New Features
```bash
# Test with sample data
python demo.py

# Test API endpoints
python main.py api
# Then visit http://localhost:8000/docs
```

## 📈 Performance Tips

### For Better Results

1. **Quality PDFs**: Use high-quality, text-based PDFs rather than scanned images
2. **Relevant Content**: Ensure your PDFs contain relevant information for your queries
3. **Appropriate Chunk Size**: Adjust `CHUNK_SIZE` based on your document structure
4. **Similarity Threshold**: Tune `SIMILARITY_THRESHOLD` for your use case

### For Better Performance

1. **Smaller Models**: Use smaller embedding models for faster processing
2. **Reduce Chunks**: Lower `TOP_K_CHUNKS` for faster retrieval
3. **Cache Results**: The system automatically caches embeddings
4. **Batch Processing**: Process multiple documents at once

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Format code: `black src/ tests/`
6. Commit: `git commit -am 'Add new feature'`
7. Push: `git push origin feature/new-feature`
8. Create a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Sentence-Transformers for multilingual embeddings
- FastAPI for the excellent web framework
- LangChain for RAG pipeline components
- Bengali NLP community for language support

## 📞 Support

- **Issues**: Create an issue in the GitHub repository
- **Documentation**: Check `/docs` when API is running
- **Examples**: See `demo.py` for usage examples
- **Testing**: Use `python main.py test` for system verification

---

**Note**: This system is designed for educational and research purposes. Ensure you have proper permissions for any PDF documents you process.
