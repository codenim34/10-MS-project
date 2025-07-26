"""
Vector Database Module for Multilingual RAG System
Handles embeddings, storage, and similarity search for Bengali and English text
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import chromadb
from chromadb.config import Settings
from dataclasses import asdict

from text_chunker import TextChunk
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Vector database for storing and retrieving document embeddings
    Supports both FAISS and ChromaDB backends
    """
    
    def __init__(self, 
                 db_path: str = Config.VECTOR_DB_PATH,
                 embedding_model: str = Config.EMBEDDING_MODEL,
                 use_chromadb: bool = True):
        self.db_path = db_path
        self.use_chromadb = use_chromadb
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize database
        if use_chromadb:
            self._init_chromadb()
        else:
            self._init_faiss()
            
        self.chunks_metadata = {}  # Store chunk metadata
        
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="multilingual_rag",
                metadata={"description": "Multilingual RAG documents with Bengali and English support"}
            )
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
            self.faiss_index = faiss.IndexIDMap(self.faiss_index)
            
            # Load existing index if available
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            metadata_path = os.path.join(self.db_path, "metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.faiss_index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    self.chunks_metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    def add_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Add text chunks to the vector database
        """
        if not chunks:
            logger.warning("No chunks to add")
            return False
            
        try:
            # Extract texts for embedding
            texts = [chunk.text for chunk in chunks]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            if self.use_chromadb:
                return self._add_to_chromadb(chunks, embeddings)
            else:
                return self._add_to_faiss(chunks, embeddings)
                
        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            return False
    
    def _add_to_chromadb(self, chunks: List[TextChunk], embeddings: np.ndarray) -> bool:
        """Add chunks to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            documents = [chunk.text for chunk in chunks]
            metadatas = [asdict(chunk) for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            
            # Convert numpy embeddings to list
            embeddings_list = embeddings.tolist()
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings_list
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to ChromaDB: {e}")
            return False
    
    def _add_to_faiss(self, chunks: List[TextChunk], embeddings: np.ndarray) -> bool:
        """Add chunks to FAISS"""
        try:
            # Generate IDs for chunks
            start_id = len(self.chunks_metadata)
            chunk_ids = list(range(start_id, start_id + len(chunks)))
            
            # Add to FAISS index
            self.faiss_index.add_with_ids(embeddings.astype(np.float32), np.array(chunk_ids))
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                self.chunks_metadata[chunk_ids[i]] = asdict(chunk)
            
            # Save index and metadata
            self._save_faiss_index()
            
            logger.info(f"Successfully added {len(chunks)} chunks to FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS: {e}")
            return False
    
    def search(self, query: str, top_k: int = Config.TOP_K_CHUNKS, threshold: float = Config.SIMILARITY_THRESHOLD) -> List[Dict[str, Any]]:
        """
        Search for similar chunks given a query
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            
            if self.use_chromadb:
                return self._search_chromadb(query, query_embedding[0], top_k, threshold)
            else:
                return self._search_faiss(query_embedding[0], top_k, threshold)
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_chromadb(self, query: str, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search using ChromaDB"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            search_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # ChromaDB returns distance, convert to similarity
                similarity = 1 - distance
                
                if similarity >= threshold:
                    search_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'similarity': similarity,
                        'rank': i + 1
                    })
            
            logger.info(f"ChromaDB search returned {len(search_results)} results above threshold {threshold}")
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        try:
            if self.faiss_index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Search in FAISS
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32), 
                min(top_k, self.faiss_index.ntotal)
            )
            
            search_results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                if similarity >= threshold:
                    metadata = self.chunks_metadata.get(int(idx), {})
                    search_results.append({
                        'text': metadata.get('text', ''),
                        'metadata': metadata,
                        'similarity': float(similarity),
                        'rank': i + 1
                    })
            
            logger.info(f"FAISS search returned {len(search_results)} results above threshold {threshold}")
            return search_results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata"""
        try:
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            metadata_path = os.path.join(self.db_path, "metadata.pkl")
            
            faiss.write_index(self.faiss_index, index_path)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
                
            logger.info("FAISS index saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database"""
        try:
            if self.use_chromadb:
                count = self.collection.count()
                return {
                    "database_type": "ChromaDB",
                    "total_chunks": count,
                    "embedding_dimension": self.embedding_dim,
                    "model": self.embedding_model
                }
            else:
                return {
                    "database_type": "FAISS",
                    "total_chunks": self.faiss_index.ntotal,
                    "embedding_dimension": self.embedding_dim,
                    "model": self.embedding_model
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def clear_database(self):
        """Clear all data from the database"""
        try:
            if self.use_chromadb:
                # Delete and recreate collection
                self.chroma_client.delete_collection("multilingual_rag")
                self.collection = self.chroma_client.get_or_create_collection(
                    name="multilingual_rag",
                    metadata={"description": "Multilingual RAG documents with Bengali and English support"}
                )
            else:
                # Reset FAISS index
                self.faiss_index.reset()
                self.chunks_metadata = {}
                self._save_faiss_index()
                
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the vector database
    from text_chunker import TextChunk
    
    # Create test chunks
    test_chunks = [
        TextChunk(
            text="অনুপমের ভাষায় সুপুরুষ বলতে শুম্ভুনাথকে বোঝানো হয়েছে।",
            metadata={"source": "test", "language": "bengali"},
            chunk_id="test_1",
            source="test_document",
            start_index=0,
            end_index=50
        ),
        TextChunk(
            text="অনুপমের মামাকে তার ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে।",
            metadata={"source": "test", "language": "bengali"},
            chunk_id="test_2",
            source="test_document",
            start_index=51,
            end_index=100
        )
    ]
    
    # Test with ChromaDB
    vector_db = VectorDatabase(use_chromadb=True)
    
    # Add chunks
    success = vector_db.add_chunks(test_chunks)
    print(f"Added chunks: {success}")
    
    # Search
    results = vector_db.search("সুপুরুষ কে?")
    print(f"Search results: {len(results)}")
    
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Similarity: {result['similarity']}")
        print("-" * 30)
