"""
Main RAG System for Multilingual Bengali-English Document QA
Combines PDF processing, text chunking, vector database, and conversation memory
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

# LLM imports
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from pdf_processor import PDFProcessor
from text_chunker import TextChunker, TextChunk
from vector_database import VectorDatabase
from conversation_memory import ConversationMemory
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualRAG:
    """
    Main RAG system that handles multilingual queries and document retrieval
    """
    
    def __init__(self, 
                 vector_db_path: str = Config.VECTOR_DB_PATH,
                 pdf_path: str = Config.PDF_PATH,
                 use_chromadb: bool = True):
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
        self.vector_db = VectorDatabase(db_path=vector_db_path, use_chromadb=use_chromadb)
        self.conversation_memory = ConversationMemory()
        
        # Initialize LLM client
        self.llm_client = None
        self.llm_type = None
        self._init_llm()
        
        # Language detection patterns
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]+')
        
        # Store paths
        self.vector_db_path = vector_db_path
        self.pdf_path = pdf_path
        
        logger.info("Multilingual RAG system initialized")
    
    def _init_llm(self):
        """Initialize LLM client based on available API keys"""
        try:
            # Try OpenAI first
            if HAS_OPENAI and Config.OPENAI_API_KEY:
                self.llm_client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self.llm_type = "openai"
                logger.info("Initialized OpenAI client")
                return
            
            # Try Google Gemini
            if HAS_GEMINI and Config.GOOGLE_API_KEY:
                genai.configure(api_key=Config.GOOGLE_API_KEY)
                self.llm_client = genai.GenerativeModel('gemini-pro')
                self.llm_type = "gemini"
                logger.info("Initialized Gemini client")
                return
            
            logger.warning("No LLM API key found. Using fallback responses.")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
    
    def setup_knowledge_base(self, pdf_files: List[str] = None) -> Dict[str, Any]:
        """
        Set up the knowledge base by processing PDF files
        """
        logger.info("Setting up knowledge base...")
        
        # If no specific files provided, process all PDFs in the directory
        if pdf_files is None:
            pdf_dir = Path(self.pdf_path)
            pdf_files = list(pdf_dir.glob("*.pdf"))
            pdf_files = [str(f) for f in pdf_files]
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_path}")
            return {"success": False, "error": "No PDF files found"}
        
        total_chunks = 0
        processed_files = []
        errors = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file}...")
                
                # Extract text from PDF
                pdf_result = self.pdf_processor.process_pdf(pdf_file)
                
                if not pdf_result["success"]:
                    errors.append(f"Failed to process {pdf_file}: {pdf_result.get('error', 'Unknown error')}")
                    continue
                
                # Chunk the text
                chunks = self.text_chunker.smart_chunk(
                    text=pdf_result["text"],
                    source=os.path.basename(pdf_file)
                )
                
                if not chunks:
                    errors.append(f"No chunks generated from {pdf_file}")
                    continue
                
                # Add chunks to vector database
                success = self.vector_db.add_chunks(chunks)
                
                if success:
                    total_chunks += len(chunks)
                    processed_files.append({
                        "file": pdf_file,
                        "chunks": len(chunks),
                        "stats": pdf_result["stats"]
                    })
                    logger.info(f"Successfully processed {pdf_file}: {len(chunks)} chunks")
                else:
                    errors.append(f"Failed to add chunks from {pdf_file} to vector database")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                errors.append(f"Error processing {pdf_file}: {str(e)}")
        
        result = {
            "success": len(processed_files) > 0,
            "total_chunks": total_chunks,
            "processed_files": processed_files,
            "errors": errors,
            "database_stats": self.vector_db.get_database_stats()
        }
        
        logger.info(f"Knowledge base setup complete: {total_chunks} chunks from {len(processed_files)} files")
        return result
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Bengali or English"""
        bengali_matches = len(self.bengali_pattern.findall(text))
        total_words = len(text.split())
        
        if total_words == 0:
            return "unknown"
        
        bengali_ratio = bengali_matches / total_words
        
        if bengali_ratio > 0.3:  # If more than 30% Bengali characters
            return "bengali"
        elif bengali_ratio > 0.1:
            return "mixed"
        else:
            return "english"
    
    def query(self, 
              user_query: str, 
              include_context: bool = True,
              top_k: int = Config.TOP_K_CHUNKS) -> Dict[str, Any]:
        """
        Main query method for the RAG system
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Detect language
            language = self.detect_language(user_query)
            
            # Retrieve relevant chunks
            retrieved_chunks = self.vector_db.search(
                query=user_query,
                top_k=top_k,
                threshold=Config.SIMILARITY_THRESHOLD
            )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found")
                fallback_response = self._get_fallback_response(user_query, language)
                
                # Store in conversation memory
                self.conversation_memory.add_conversation_turn(
                    user_query=user_query,
                    system_response=fallback_response,
                    language=language,
                    retrieved_chunks=[],
                    metadata={"type": "fallback", "chunks_found": 0}
                )
                
                return {
                    "query": user_query,
                    "response": fallback_response,
                    "language": language,
                    "retrieved_chunks": [],
                    "conversation_context": "",
                    "metadata": {
                        "chunks_found": 0,
                        "type": "fallback"
                    }
                }
            
            # Get conversation context if needed
            conversation_context = ""
            if include_context:
                conversation_context = self.conversation_memory.format_context_for_llm(3)
            
            # Generate response using LLM
            response = self._generate_response(
                query=user_query,
                retrieved_chunks=retrieved_chunks,
                conversation_context=conversation_context,
                language=language
            )
            
            # Store in conversation memory
            self.conversation_memory.add_conversation_turn(
                user_query=user_query,
                system_response=response,
                language=language,
                retrieved_chunks=retrieved_chunks,
                metadata={
                    "chunks_found": len(retrieved_chunks),
                    "type": "rag_response",
                    "avg_similarity": sum(chunk["similarity"] for chunk in retrieved_chunks) / len(retrieved_chunks)
                }
            )
            
            return {
                "query": user_query,
                "response": response,
                "language": language,
                "retrieved_chunks": retrieved_chunks,
                "conversation_context": conversation_context,
                "metadata": {
                    "chunks_found": len(retrieved_chunks),
                    "type": "rag_response",
                    "avg_similarity": sum(chunk["similarity"] for chunk in retrieved_chunks) / len(retrieved_chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            error_response = f"দুঃখিত, আমি এই প্রশ্নের উত্তর দিতে পারছি না। / Sorry, I cannot answer this question right now."
            
            return {
                "query": user_query,
                "response": error_response,
                "language": self.detect_language(user_query),
                "retrieved_chunks": [],
                "conversation_context": "",
                "metadata": {"type": "error", "error": str(e)}
            }
    
    def _generate_response(self, 
                          query: str, 
                          retrieved_chunks: List[Dict[str, Any]], 
                          conversation_context: str,
                          language: str) -> str:
        """Generate response using LLM"""
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"Context {i+1} (similarity: {chunk['similarity']:.3f}):")
            context_parts.append(chunk['text'])
            context_parts.append("")
        
        context_text = "\n".join(context_parts)
        
        # Create prompt based on language
        if language == "bengali" or language == "mixed":
            system_prompt = """আপনি একজন সহায়ক AI সহায়ক যিনি বাংলা এবং ইংরেজি উভয় ভাষায় উত্তর দিতে পারেন। 
            আপনাকে প্রদান করা প্রসঙ্গের উপর ভিত্তি করে প্রশ্নের উত্তর দিন। যদি প্রসঙ্গে উত্তর না থাকে, তাহলে বলুন যে আপনি জানেন না।
            উত্তর সংক্ষিপ্ত এবং সঠিক হতে হবে।"""
            
            prompt = f"""{system_prompt}

পূর্ববর্তী কথোপকথন:
{conversation_context}

প্রসঙ্গ:
{context_text}

প্রশ্ন: {query}

উত্তর:"""
        else:
            system_prompt = """You are a helpful AI assistant that can answer questions in both Bengali and English.
            Please answer the question based on the provided context. If the context doesn't contain the answer, say you don't know.
            Keep your answer concise and accurate."""
            
            prompt = f"""{system_prompt}

Previous conversation:
{conversation_context}

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Generate response using available LLM
        if self.llm_client and self.llm_type == "openai":
            try:
                response = self.llm_client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
        
        elif self.llm_client and self.llm_type == "gemini":
            try:
                response = self.llm_client.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
        
        # Fallback to rule-based response
        return self._get_rule_based_response(query, retrieved_chunks, language)
    
    def _get_rule_based_response(self, query: str, chunks: List[Dict[str, Any]], language: str) -> str:
        """Generate rule-based response when LLM is not available"""
        
        if not chunks:
            if language == "bengali":
                return "দুঃখিত, আমি এই প্রশ্নের উত্তর খুঁজে পাইনি।"
            else:
                return "Sorry, I couldn't find an answer to this question."
        
        # Find the most relevant chunk
        best_chunk = chunks[0]
        
        # Extract potential answer using simple heuristics
        chunk_text = best_chunk['text']
        
        # Simple pattern matching for Bengali questions
        if "কাকে" in query or "কে" in query:  # Who questions
            # Look for names in the chunk
            bengali_names = re.findall(r'[\u0980-\u09FF]+(?:[\u0980-\u09FF\s])*[\u0980-\u09FF]+', chunk_text)
            if bengali_names:
                if language == "bengali":
                    return f"প্রদত্ত তথ্য অনুযায়ী: {bengali_names[0]}"
                else:
                    return f"According to the provided information: {bengali_names[0]}"
        
        # Default response with the most relevant chunk
        if language == "bengali":
            return f"সংশ্লিষ্ট তথ্য: {chunk_text[:200]}..."
        else:
            return f"Related information: {chunk_text[:200]}..."
    
    def _get_fallback_response(self, query: str, language: str) -> str:
        """Get fallback response when no chunks are found"""
        if language == "bengali":
            return "দুঃখিত, আমি এই প্রশ্নের সাথে সম্পর্কিত কোন তথ্য খুঁজে পাইনি। আপনি কি অন্যভাবে প্রশ্নটি করতে পারেন?"
        else:
            return "Sorry, I couldn't find any information related to this question. Could you try rephrasing your question?"
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            db_stats = self.vector_db.get_database_stats()
            memory_stats = self.conversation_memory.get_conversation_stats()
            
            return {
                "vector_database": db_stats,
                "conversation_memory": memory_stats,
                "llm_type": self.llm_type,
                "config": {
                    "chunk_size": Config.CHUNK_SIZE,
                    "chunk_overlap": Config.CHUNK_OVERLAP,
                    "top_k_chunks": Config.TOP_K_CHUNKS,
                    "similarity_threshold": Config.SIMILARITY_THRESHOLD
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_memory.clear_all_memory()
        logger.info("Conversation history cleared")
    
    def export_conversation_history(self, output_path: str, language: str = None) -> bool:
        """Export conversation history to file"""
        return self.conversation_memory.export_conversations(output_path, language)

# Example usage and testing
if __name__ == "__main__":
    # Initialize RAG system
    rag = MultilingualRAG()
    
    # Set up knowledge base (you need to add PDF files to data/pdfs/)
    print("Setting up knowledge base...")
    setup_result = rag.setup_knowledge_base()
    print(f"Setup result: {setup_result}")
    
    # Test queries
    test_queries = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "Who is mentioned as a good person according to Anupam?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        print(f"Response: {result['response']}")
        print(f"Language: {result['language']}")
        print(f"Chunks found: {result['metadata']['chunks_found']}")
        print("-" * 50)
