"""
Unit Tests for PDF Processor Module
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.pdf_processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = PDFProcessor()
    
    def test_bengali_pattern(self):
        """Test Bengali character detection"""
        bengali_text = "অনুপমের ভাষায় সুপুরুষ"
        english_text = "This is English text"
        mixed_text = "This has অনুপম Bengali"
        
        bengali_matches = self.processor.bengali_pattern.findall(bengali_text)
        english_matches = self.processor.bengali_pattern.findall(english_text)
        mixed_matches = self.processor.bengali_pattern.findall(mixed_text)
        
        self.assertTrue(len(bengali_matches) > 0)
        self.assertEqual(len(english_matches), 0)
        self.assertTrue(len(mixed_matches) > 0)
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        dirty_text = "অনুপমের    ভাষায়।।   সুপুরুষ\n\n\nকাকে"
        clean_text = self.processor.clean_text(dirty_text)
        
        self.assertNotIn("।।", clean_text)
        self.assertNotIn("   ", clean_text)
        self.assertTrue(len(clean_text) > 0)
    
    def test_generate_stats(self):
        """Test statistics generation"""
        text = "অনুপমের ভাষায় সুপুরুষ। This is English text."
        stats = self.processor.generate_stats(text)
        
        self.assertIn('total_characters', stats)
        self.assertIn('bengali_words', stats)
        self.assertIn('has_bengali', stats)
        self.assertTrue(stats['has_bengali'])
        self.assertTrue(stats['bengali_percentage'] > 0)
    
    @patch('src.pdf_processor.fitz.open')
    def test_extract_text_pymupdf_success(self, mock_fitz_open):
        """Test successful PyMuPDF extraction"""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text from PDF"
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_fitz_open.return_value = mock_doc
        
        result = self.processor.extract_text_pymupdf("dummy.pdf")
        
        self.assertEqual(result, "Sample text from PDF\n")
        mock_doc.close.assert_called_once()
    
    @patch('src.pdf_processor.fitz.open')
    def test_extract_text_pymupdf_failure(self, mock_fitz_open):
        """Test PyMuPDF extraction failure"""
        mock_fitz_open.side_effect = Exception("PDF read error")
        
        result = self.processor.extract_text_pymupdf("dummy.pdf")
        
        self.assertEqual(result, "")
    
    def test_process_pdf_nonexistent_file(self):
        """Test processing non-existent file"""
        result = self.processor.process_pdf("nonexistent.pdf")
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)

if __name__ == '__main__':
    unittest.main()
