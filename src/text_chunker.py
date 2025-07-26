"""
Text Chunking Module for Multilingual RAG System
Implements various chunking strategies optimized for Bengali and English text
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Data class representing a text chunk"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    start_index: int
    end_index: int

class TextChunker:
    """
    Advanced text chunking system with multiple strategies
    Optimized for multilingual content including Bengali
    """
    
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Bengali sentence endings
        self.bengali_sentence_endings = ['।', '?', '!', ':', ';']
        # English sentence endings
        self.english_sentence_endings = ['.', '?', '!', ':', ';']
        
        # Combined pattern for sentence detection
        self.sentence_pattern = re.compile(r'[।.!?:;]+')
        
        # Bengali word boundary pattern
        self.bengali_word_pattern = re.compile(r'[\u0980-\u09FF]+')
        
    def chunk_by_sentences(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Chunk text by sentences, respecting both Bengali and English sentence boundaries
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return chunks
            
        current_chunk = ""
        current_start = 0
        chunk_count = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        metadata={
                            "chunk_method": "sentence-based",
                            "sentence_count": len(self._split_into_sentences(current_chunk)),
                            "has_bengali": bool(self.bengali_word_pattern.search(current_chunk)),
                            "language_ratio": self._calculate_language_ratio(current_chunk)
                        },
                        chunk_id=f"{source}_chunk_{chunk_count}",
                        source=source,
                        start_index=current_start,
                        end_index=chunk_end
                    )
                    chunks.append(chunk)
                    chunk_count += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(sentences, i, current_chunk)
                current_chunk = overlap_sentences + " " + sentence if overlap_sentences else sentence
                current_start = chunk_end - len(overlap_sentences) if overlap_sentences else chunk_end
        
        # Add the last chunk
        if current_chunk:
            chunk = TextChunk(
                text=current_chunk.strip(),
                metadata={
                    "chunk_method": "sentence-based",
                    "sentence_count": len(self._split_into_sentences(current_chunk)),
                    "has_bengali": bool(self.bengali_word_pattern.search(current_chunk)),
                    "language_ratio": self._calculate_language_ratio(current_chunk)
                },
                chunk_id=f"{source}_chunk_{chunk_count}",
                source=source,
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            )
            chunks.append(chunk)
            
        return chunks
    
    def chunk_by_paragraphs(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Chunk text by paragraphs, combining small paragraphs
        """
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        chunk_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    chunk = TextChunk(
                        text=current_chunk.strip(),
                        metadata={
                            "chunk_method": "paragraph-based",
                            "paragraph_count": len(current_chunk.split('\n\n')),
                            "has_bengali": bool(self.bengali_word_pattern.search(current_chunk)),
                            "language_ratio": self._calculate_language_ratio(current_chunk)
                        },
                        chunk_id=f"{source}_chunk_{chunk_count}",
                        source=source,
                        start_index=current_start,
                        end_index=chunk_end
                    )
                    chunks.append(chunk)
                    chunk_count += 1
                
                # Start new chunk with current paragraph
                current_chunk = paragraph
                current_start = chunk_end
        
        # Add the last chunk
        if current_chunk:
            chunk = TextChunk(
                text=current_chunk.strip(),
                metadata={
                    "chunk_method": "paragraph-based",
                    "paragraph_count": len(current_chunk.split('\n\n')),
                    "has_bengali": bool(self.bengali_word_pattern.search(current_chunk)),
                    "language_ratio": self._calculate_language_ratio(current_chunk)
                },
                chunk_id=f"{source}_chunk_{chunk_count}",
                source=source,
                start_index=current_start,
                end_index=current_start + len(current_chunk)
            )
            chunks.append(chunk)
            
        return chunks
    
    def chunk_by_fixed_size(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Chunk text by fixed character size with overlap
        """
        chunks = []
        chunk_count = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            # Try to break at word boundary for better readability
            if i + self.chunk_size < len(text):
                # Find last space to avoid breaking words
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size * 0.8:  # Only if not too far back
                    chunk_text = chunk_text[:last_space]
            
            chunk = TextChunk(
                text=chunk_text.strip(),
                metadata={
                    "chunk_method": "fixed-size",
                    "original_size": len(chunk_text),
                    "has_bengali": bool(self.bengali_word_pattern.search(chunk_text)),
                    "language_ratio": self._calculate_language_ratio(chunk_text)
                },
                chunk_id=f"{source}_chunk_{chunk_count}",
                source=source,
                start_index=i,
                end_index=i + len(chunk_text)
            )
            chunks.append(chunk)
            chunk_count += 1
            
        return chunks
    
    def smart_chunk(self, text: str, source: str = "") -> List[TextChunk]:
        """
        Smart chunking that adapts based on text characteristics
        """
        # Analyze text characteristics
        analysis = self._analyze_text(text)
        
        if analysis["avg_paragraph_length"] > self.chunk_size * 1.5:
            # Long paragraphs - use sentence-based chunking
            logger.info("Using sentence-based chunking for long paragraphs")
            return self.chunk_by_sentences(text, source)
        elif analysis["paragraph_count"] > 10 and analysis["avg_paragraph_length"] < self.chunk_size * 0.5:
            # Many short paragraphs - use paragraph-based chunking
            logger.info("Using paragraph-based chunking for short paragraphs")
            return self.chunk_by_paragraphs(text, source)
        else:
            # Default to sentence-based chunking
            logger.info("Using default sentence-based chunking")
            return self.chunk_by_sentences(text, source)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences handling both Bengali and English"""
        # Split by sentence endings
        sentences = re.split(r'[।.!?:;]+', text)
        
        # Clean and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences
    
    def _get_overlap_sentences(self, sentences: List[str], current_index: int, current_chunk: str) -> str:
        """Get overlap sentences for chunk continuity"""
        if self.chunk_overlap <= 0:
            return ""
            
        chunk_sentences = self._split_into_sentences(current_chunk)
        overlap_count = min(2, len(chunk_sentences))  # Max 2 sentences overlap
        
        if overlap_count > 0:
            return " ".join(chunk_sentences[-overlap_count:])
        return ""
    
    def _calculate_language_ratio(self, text: str) -> Dict[str, float]:
        """Calculate the ratio of Bengali to English content"""
        words = text.split()
        bengali_words = len(self.bengali_word_pattern.findall(text))
        english_words = len(words) - bengali_words
        
        total_words = len(words)
        if total_words == 0:
            return {"bengali": 0.0, "english": 0.0}
            
        return {
            "bengali": bengali_words / total_words,
            "english": english_words / total_words
        }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics for smart chunking"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return {
                "paragraph_count": 0,
                "avg_paragraph_length": 0,
                "total_length": len(text)
            }
        
        paragraph_lengths = [len(p) for p in paragraphs]
        
        return {
            "paragraph_count": len(paragraphs),
            "avg_paragraph_length": sum(paragraph_lengths) / len(paragraph_lengths),
            "min_paragraph_length": min(paragraph_lengths),
            "max_paragraph_length": max(paragraph_lengths),
            "total_length": len(text)
        }

# Example usage
if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    
    sample_text = """
    অনুপমের ভাষায় সুপুরুষ বলতে শুম্ভুনাথকে বোঝানো হয়েছে। তিনি ছিলেন একজন আদর্শ পুরুষ।
    
    অনুপমের মামাকে তার ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। মামা তার জীবনে গুরুত্বপূর্ণ ভূমিকা পালন করেছেন।
    
    কল্যাণীর বিয়ের সময় তার বয়স ছিল মাত্র ১৫ বছর। এটি ছিল সেই সময়ের একটি সাধারণ বিষয়।
    """
    
    chunks = chunker.smart_chunk(sample_text, "test_document")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"ID: {chunk.chunk_id}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)
