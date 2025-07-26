"""
Unit Tests for Text Chunker Module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.text_chunker import TextChunker, TextChunk

class TestTextChunker(unittest.TestCase):
    
    def setUp(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    def test_bengali_language_detection(self):
        """Test language ratio calculation"""
        bengali_text = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে"
        english_text = "This is a simple English sentence"
        mixed_text = "অনুপম is a Bengali name"
        
        bengali_ratio = self.chunker._calculate_language_ratio(bengali_text)
        english_ratio = self.chunker._calculate_language_ratio(english_text)
        mixed_ratio = self.chunker._calculate_language_ratio(mixed_text)
        
        self.assertTrue(bengali_ratio["bengali"] > 0.8)
        self.assertTrue(english_ratio["english"] > 0.8)
        self.assertTrue(mixed_ratio["bengali"] > 0 and mixed_ratio["english"] > 0)
    
    def test_sentence_splitting(self):
        """Test sentence splitting for Bengali and English"""
        text = "অনুপম একজন ভাল ছেলে। তিনি পড়াশোনায় ভাল। He is good at studies. What about others?"
        sentences = self.chunker._split_into_sentences(text)
        
        self.assertTrue(len(sentences) >= 3)
        self.assertTrue(any("অনুপম" in s for s in sentences))
        self.assertTrue(any("studies" in s for s in sentences))
    
    def test_chunk_by_sentences(self):
        """Test sentence-based chunking"""
        text = """অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে। তাকে শুম্ভুনাথ বলা হয়েছে। 
        কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে। তাকে মামা বলা হয়েছে।
        বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল। তার বয়স ছিল ১৫ বছর।"""
        
        chunks = self.chunker.chunk_by_sentences(text, "test_doc")
        
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertTrue(len(chunk.text) <= self.chunker.chunk_size or chunks.index(chunk) == 0)
            self.assertEqual(chunk.source, "test_doc")
            self.assertTrue(chunk.metadata["has_bengali"])
    
    def test_chunk_by_paragraphs(self):
        """Test paragraph-based chunking"""
        text = """অনুপমের প্রথম অনুচ্ছেদ। এখানে কিছু তথ্য আছে।

দ্বিতীয় অনুচ্ছেদ এখানে। আরও কিছু বিষয় নিয়ে।

তৃতীয় অনুচ্ছেদ। শেষ কিছু কথা।"""
        
        chunks = self.chunker.chunk_by_paragraphs(text, "test_doc")
        
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertEqual(chunk.metadata["chunk_method"], "paragraph-based")
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking"""
        text = "অ" * 500  # Long Bengali text
        
        chunks = self.chunker.chunk_by_fixed_size(text, "test_doc")
        
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, TextChunk)
            self.assertTrue(len(chunk.text) <= self.chunker.chunk_size)
            self.assertEqual(chunk.metadata["chunk_method"], "fixed-size")
    
    def test_smart_chunking(self):
        """Test smart chunking strategy selection"""
        # Short paragraphs - should use paragraph-based
        short_para_text = "প্রথম।\n\nদ্বিতীয়।\n\nতৃতীয়।\n\nচতুর্থ।"
        
        # Long paragraph - should use sentence-based  
        long_para_text = "অনুপম একজন ভাল ছেলে। " * 20
        
        short_chunks = self.chunker.smart_chunk(short_para_text, "short_test")
        long_chunks = self.chunker.smart_chunk(long_para_text, "long_test")
        
        self.assertTrue(len(short_chunks) > 0)
        self.assertTrue(len(long_chunks) > 0)
    
    def test_text_analysis(self):
        """Test text analysis for smart chunking"""
        text = """অনুচ্ছেদ ১। কিছু বিষয়।

অনুচ্ছেদ ২। আরও বিষয়।

অনুচ্ছেদ ৩। শেষ বিষয়।"""
        
        analysis = self.chunker._analyze_text(text)
        
        self.assertIn("paragraph_count", analysis)
        self.assertIn("avg_paragraph_length", analysis)
        self.assertTrue(analysis["paragraph_count"] > 0)
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text"""
        empty_text = ""
        whitespace_text = "   \n\n   "
        
        empty_chunks = self.chunker.smart_chunk(empty_text, "empty_test")
        whitespace_chunks = self.chunker.smart_chunk(whitespace_text, "whitespace_test")
        
        self.assertEqual(len(empty_chunks), 0)
        self.assertEqual(len(whitespace_chunks), 0)

if __name__ == '__main__':
    unittest.main()
