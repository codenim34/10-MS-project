"""
Main Application Entry Point for Multilingual RAG System
Provides CLI interface and application orchestration
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from src.multilingual_rag import MultilingualRAG
from src.evaluation import run_evaluation, DEFAULT_TEST_CASES
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        Config.PDF_PATH,
        Config.VECTOR_DB_PATH,
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_knowledge_base(rag_system: MultilingualRAG, pdf_files: list = None):
    """Setup knowledge base from PDF files"""
    print("Setting up knowledge base...")
    
    if pdf_files:
        # Use specified files
        result = rag_system.setup_knowledge_base(pdf_files)
    else:
        # Use all PDFs in the directory
        result = rag_system.setup_knowledge_base()
    
    if result["success"]:
        print(f"✅ Knowledge base setup successful!")
        print(f"   Total chunks: {result['total_chunks']}")
        print(f"   Processed files: {len(result['processed_files'])}")
        
        for file_info in result['processed_files']:
            print(f"   - {file_info['file']}: {file_info['chunks']} chunks")
    else:
        print("❌ Knowledge base setup failed!")
        for error in result["errors"]:
            print(f"   Error: {error}")
    
    return result

def interactive_query_session(rag_system: MultilingualRAG):
    """Interactive query session"""
    print("\n🤖 Multilingual RAG System - Interactive Session")
    print("Ask questions in Bengali or English. Type 'quit' to exit.")
    print("Type 'stats' to see system statistics.")
    print("Type 'clear' to clear conversation history.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n📝 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = rag_system.get_system_stats()
                print("\n📊 System Statistics:")
                db_stats = stats.get('vector_database', {})
                print(f"   Total chunks: {db_stats.get('total_chunks', 0)}")
                print(f"   Database type: {db_stats.get('database_type', 'Unknown')}")
                print(f"   LLM type: {stats.get('llm_type', 'None')}")
                continue
            
            if query.lower() == 'clear':
                rag_system.clear_conversation_history()
                print("🗑️ Conversation history cleared!")
                continue
            
            print("🔍 Processing your question...")
            
            # Process the query
            result = rag_system.query(query)
            
            print(f"\n🤖 Answer ({result['language']}):")
            print(f"   {result['response']}")
            
            if result['retrieved_chunks']:
                print(f"\n📚 Found {len(result['retrieved_chunks'])} relevant chunks:")
                for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                    print(f"   {i}. Similarity: {chunk['similarity']:.3f}")
                    print(f"      Text: {chunk['text'][:100]}...")
            else:
                print("   No relevant chunks found in knowledge base.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_test_queries(rag_system: MultilingualRAG):
    """Run predefined test queries"""
    print("\n🧪 Running Test Queries...")
    print("-" * 50)
    
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "Who is mentioned as a good person according to Anupam?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        try:
            result = rag_system.query(query)
            print(f"   Answer: {result['response']}")
            print(f"   Language: {result['language']}")
            print(f"   Chunks found: {result['metadata']['chunks_found']}")
            
            if result['retrieved_chunks']:
                avg_similarity = sum(chunk['similarity'] for chunk in result['retrieved_chunks']) / len(result['retrieved_chunks'])
                print(f"   Avg similarity: {avg_similarity:.3f}")
                
        except Exception as e:
            print(f"   Error: {e}")

def run_evaluation_cmd(rag_system: MultilingualRAG, output_path: str = None):
    """Run system evaluation"""
    print("\n📊 Running System Evaluation...")
    print("-" * 50)
    
    if output_path is None:
        output_path = "data/evaluation_report.json"
    
    try:
        results = run_evaluation(rag_system, output_path=output_path)
        
        print("✅ Evaluation completed!")
        print(f"   Report saved to: {output_path}")
        
        # Print summary
        summary = results['evaluation_summary']
        print(f"\n📈 Evaluation Summary:")
        print(f"   Total queries: {summary['total_queries']}")
        print(f"   Languages: {list(summary['language_distribution'].keys())}")
        
        overall_perf = summary.get('overall_performance', {})
        for metric in ['semantic_similarity', 'groundedness', 'relevance', 'exact_match']:
            if metric in overall_perf:
                score = overall_perf[metric]['mean']
                print(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
                
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

def start_api_server():
    """Start the REST API server"""
    print("\n🚀 Starting REST API Server...")
    print(f"   Host: {Config.API_HOST}")
    print(f"   Port: {Config.API_PORT}")
    print(f"   Docs: http://{Config.API_HOST}:{Config.API_PORT}/docs")
    print("-" * 50)
    
    try:
        from src.api import run_server
        run_server()
    except ImportError as e:
        print(f"❌ Failed to import API module: {e}")
        print("   Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Multilingual RAG System for Bengali-English Document QA"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup knowledge base from PDFs')
    setup_parser.add_argument('--pdf-files', nargs='+', help='Specific PDF files to process')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Interactive query session')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test queries')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run system evaluation')
    eval_parser.add_argument('--output', help='Output file for evaluation report')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup directories
    setup_directories()
    
    # Initialize RAG system
    print("🤖 Initializing Multilingual RAG System...")
    try:
        rag_system = MultilingualRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Execute command
    if args.command == 'setup':
        setup_knowledge_base(rag_system, args.pdf_files)
    
    elif args.command == 'query':
        interactive_query_session(rag_system)
    
    elif args.command == 'test':
        run_test_queries(rag_system)
    
    elif args.command == 'evaluate':
        run_evaluation_cmd(rag_system, args.output)
    
    elif args.command == 'api':
        start_api_server()
    
    elif args.command == 'status':
        try:
            stats = rag_system.get_system_stats()
            print("\n📊 System Status:")
            print(f"   Database: {stats.get('vector_database', {}).get('database_type', 'Unknown')}")
            print(f"   Total chunks: {stats.get('vector_database', {}).get('total_chunks', 0)}")
            print(f"   LLM: {stats.get('llm_type', 'None configured')}")
            
            memory_stats = stats.get('conversation_memory', {})
            short_term = memory_stats.get('short_term_memory', {})
            long_term = memory_stats.get('long_term_memory', {})
            
            print(f"   Short-term memory: {short_term.get('current_turns', 0)}/{short_term.get('max_capacity', 0)}")
            print(f"   Long-term conversations: {long_term.get('total_conversations', 0)}")
            
        except Exception as e:
            print(f"❌ Failed to get system status: {e}")

if __name__ == "__main__":
    main()
