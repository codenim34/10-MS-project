"""
Demo Script for Multilingual RAG System
Demonstrates the complete workflow with sample data
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.multilingual_rag import MultilingualRAG
from src.evaluation import run_evaluation

def create_sample_text_file():
    """Create a sample text file with Bengali content for testing"""
    sample_content = """
অনুপম চরিত্র বিশ্লেষণ

অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? অনুপমের ভাষায় সুপুরুষ বলতে শুম্ভুনাথকে বোঝানো হয়েছে। শুম্ভুনাথ একজন আদর্শ পুরুষ হিসেবে চিত্রিত হয়েছেন।

কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? অনুপমের মামাকে তার ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। মামা অনুপমের জীবনে গুরুত্বপূর্ণ ভূমিকা পালন করেছেন।

বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? বিয়ের সময় কল্যাণীর প্রকৃত বয়স ছিল মাত্র ১৫ বছর। এটি সেই সময়ের সামাজিক প্রথার একটি উদাহরণ।

চরিত্র বিশ্লেষণ:
অনুপম একজন শিক্ষিত যুবক যিনি তার সময়ের সামাজিক সমস্যা নিয়ে চিন্তা করেন। তিনি বিবাহের ক্ষেত্রে যৌতুক প্রথার বিরোধী।

কল্যাণী একজন তরুণী যার বিয়ে অল্প বয়সে হয়েছিল। তার চরিত্রে সেই সময়ের নারীদের অবস্থান প্রতিফলিত হয়েছে।

শুম্ভুনাথ অনুপমের আদর্শ পুরুষের প্রতিমূর্তি। তিনি শিক্ষিত, চরিত্রবান এবং সমাজ সচেতন।

সামাজিক প্রেক্ষাপট:
এই গল্পে বাংলার সামাজিক রীতি-নীতি, বিবাহ প্রথা এবং শিক্ষিত মধ্যবিত্ত সমাজের চিত্র ফুটে উঠেছে।
"""
    
    # Create data directory if it doesn't exist
    os.makedirs("data/pdfs", exist_ok=True)
    
    # Write sample content to a text file (simulating PDF content)
    with open("data/sample_bengali_content.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    return "data/sample_bengali_content.txt"

def demo_text_processing():
    """Demonstrate text processing capabilities"""
    print("📚 Demo: Text Processing Capabilities")
    print("=" * 50)
    
    # Create sample content
    sample_file = create_sample_text_file()
    print(f"✅ Created sample content: {sample_file}")
    
    # Initialize components
    from src.pdf_processor import PDFProcessor
    from src.text_chunker import TextChunker
    
    processor = PDFProcessor()
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    
    # Read and process sample content
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    # Generate statistics
    stats = processor.generate_stats(sample_text)
    print(f"\n📊 Text Statistics:")
    print(f"   Total characters: {stats['total_characters']}")
    print(f"   Total words: {stats['total_words']}")
    print(f"   Bengali words: {stats['bengali_words']}")
    print(f"   Bengali percentage: {stats['bengali_percentage']:.1f}%")
    
    # Demonstrate chunking
    chunks = chunker.smart_chunk(sample_text, "sample_document")
    print(f"\n✂️ Text Chunking:")
    print(f"   Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n   Chunk {i}:")
        print(f"     Method: {chunk.metadata.get('chunk_method', 'unknown')}")
        print(f"     Length: {len(chunk.text)} characters")
        print(f"     Has Bengali: {chunk.metadata.get('has_bengali', False)}")
        print(f"     Text preview: {chunk.text[:100]}...")

def demo_rag_system():
    """Demonstrate RAG system capabilities"""
    print("🤖 Demo: RAG System Capabilities")  
    print("=" * 50)
    
    # Initialize RAG system
    rag = MultilingualRAG()
    
    # Create sample content for knowledge base
    sample_file = create_sample_text_file()
    
    # Simulate PDF processing by directly adding text
    from src.text_chunker import TextChunker
    chunker = TextChunker()
    
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    chunks = chunker.smart_chunk(sample_text, "sample_document")
    
    # Add chunks to vector database
    print("🔄 Setting up knowledge base...")
    success = rag.vector_db.add_chunks(chunks)
    
    if success:
        print("✅ Knowledge base setup successful!")
        
        # Test queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", 
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "Who is described as a good person by Anupam?"
        ]
        
        print(f"\n❓ Testing {len(test_queries)} queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            
            try:
                result = rag.query(query)
                print(f"   Language: {result['language']}")
                print(f"   Response: {result['response']}")
                print(f"   Chunks found: {result['metadata']['chunks_found']}")
                
                if result['retrieved_chunks']:
                    avg_sim = sum(c['similarity'] for c in result['retrieved_chunks']) / len(result['retrieved_chunks'])
                    print(f"   Avg similarity: {avg_sim:.3f}")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        print("❌ Knowledge base setup failed!")

def demo_api_usage():
    """Demonstrate API usage with sample requests"""
    print("🌐 Demo: API Usage Examples")
    print("=" * 50)
    
    print("To test the API, run these commands in separate terminals:")
    print()
    print("1. Start the API server:")
    print("   python main.py api")
    print()
    print("2. Test health endpoint:")
    print('   curl http://localhost:8000/health')
    print()
    print("3. Test query endpoint:")
    print('   curl -X POST "http://localhost:8000/query" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}\'')
    print()
    print("4. Get system stats:")
    print('   curl http://localhost:8000/stats')
    print()
    print("5. Access interactive documentation:")
    print("   Open http://localhost:8000/docs in your browser")

def demo_evaluation():
    """Demonstrate evaluation capabilities"""
    print("📊 Demo: Evaluation System")
    print("=" * 50)
    
    # Create sample content
    sample_file = create_sample_text_file()
    
    try:
        # Initialize RAG system with sample data
        rag = MultilingualRAG()
        
        # Add sample content to knowledge base
        from src.text_chunker import TextChunker
        chunker = TextChunker()
        
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_text = f.read()
        
        chunks = chunker.smart_chunk(sample_text, "sample_document")
        success = rag.vector_db.add_chunks(chunks)
        
        if success:
            print("✅ Knowledge base ready for evaluation")
            
            # Sample test cases
            test_cases = [
                {
                    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                    "expected_answer": "শুম্ভুনাথ" 
                },
                {
                    "query": "কল্যাণীর বয়স কত ছিল?",
                    "expected_answer": "১৫ বছর"
                }
            ]
            
            print(f"\n🧪 Running evaluation with {len(test_cases)} test cases...")
            
            from src.evaluation import RAGEvaluator
            evaluator = RAGEvaluator(rag)
            
            results = evaluator.evaluate_test_set(test_cases)
            
            print("📈 Evaluation Results:")
            summary = results['evaluation_summary']
            
            print(f"   Total queries: {summary['total_queries']}")
            print(f"   Languages tested: {list(summary['language_distribution'].keys())}")
            
            # Show individual results
            for i, result in enumerate(results['individual_results'], 1):
                print(f"\n   Test {i}:")
                print(f"     Query: {result['query']}")
                print(f"     Expected: {result['ground_truth']}")
                print(f"     Got: {result['predicted_answer']}")
                print(f"     Similarity: {result['metrics'].get('semantic_similarity', 0):.3f}")
                print(f"     Groundedness: {result['metrics'].get('groundedness', 0):.3f}")
        else:
            print("❌ Failed to set up knowledge base for evaluation")
            
    except Exception as e:
        print(f"❌ Evaluation demo failed: {e}")

def main():
    """Run all demonstrations"""
    print("🚀 Multilingual RAG System - Complete Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run demonstrations
    try:
        demo_text_processing()
        print("\n" + "=" * 60 + "\n")
        
        demo_rag_system()
        print("\n" + "=" * 60 + "\n")
        
        demo_api_usage()
        print("\n" + "=" * 60 + "\n")
        
        demo_evaluation()
        print("\n" + "=" * 60 + "\n")
        
        print("✅ All demonstrations completed successfully!")
        print()
        print("Next steps:")
        print("1. Add your Bengali PDF files to data/pdfs/")
        print("2. Set up your API keys in .env file")
        print("3. Run 'python main.py setup' to build knowledge base")
        print("4. Start interactive session with 'python main.py query'")
        print("5. Launch API server with 'python main.py api'")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
