"""
PDF Processing Module for Multilingual RAG System
Handles extraction and preprocessing of Bengali text from PDF documents
"""

import PyPDF2
import fitz  # PyMuPDF
import pdfplumber
import re
import logging
from typing import List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    A comprehensive PDF processor that handles multilingual text extraction
    with special focus on Bengali language support.
    """
    
    def __init__(self):
        self.bengali_pattern = re.compile(r'[\u0980-\u09FF]+')  # Bengali Unicode range
        
    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 - good for simple PDFs"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF - better for complex layouts"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber - best for table extraction"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
    
    def extract_text_hybrid(self, pdf_path: str) -> str:
        """
        Hybrid approach - tries multiple extraction methods and returns the best result
        """
        methods = [
            ("PyMuPDF", self.extract_text_pymupdf),
            ("pdfplumber", self.extract_text_pdfplumber),
            ("PyPDF2", self.extract_text_pypdf2)
        ]
        
        best_text = ""
        best_score = 0
        
        for method_name, method in methods:
            try:
                text = method(pdf_path)
                # Score based on Bengali content and total length
                bengali_matches = len(self.bengali_pattern.findall(text))
                score = len(text) + bengali_matches * 10  # Bonus for Bengali content
                
                logger.info(f"{method_name}: {len(text)} chars, {bengali_matches} Bengali matches, score: {score}")
                
                if score > best_score:
                    best_text = text
                    best_score = score
                    
            except Exception as e:
                logger.error(f"{method_name} failed: {e}")
                
        return best_text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\nPage \d+\n', '\n', text)
        
        # Fix common OCR issues in Bengali
        # Replace common misrecognized characters
        replacements = {
            'া।': 'া।',  # Fix punctuation
            '।।': '।',   # Fix double punctuation
            'ঃঃ': 'ঃ',   # Fix double colons
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive punctuation
        text = re.sub(r'([।,;]){2,}', r'\1', text)
        
        # Clean up whitespace around Bengali punctuation
        text = re.sub(r' +।', '।', text)
        text = re.sub(r'। +', '। ', text)
        
        return text.strip()
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method to process a PDF file
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text using hybrid approach
        raw_text = self.extract_text_hybrid(pdf_path)
        
        if not raw_text:
            logger.error(f"Failed to extract text from {pdf_path}")
            return {
                "success": False,
                "error": "Text extraction failed",
                "text": "",
                "stats": {}
            }
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Generate statistics
        stats = self.generate_stats(cleaned_text)
        
        logger.info(f"Successfully processed {pdf_path}: {stats}")
        
        return {
            "success": True,
            "text": cleaned_text,
            "stats": stats,
            "file_path": pdf_path
        }
    
    def generate_stats(self, text: str) -> Dict[str, Any]:
        """Generate statistics about the processed text"""
        bengali_matches = self.bengali_pattern.findall(text)
        
        return {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "bengali_words": len(bengali_matches),
            "paragraphs": len(text.split('\n\n')),
            "has_bengali": len(bengali_matches) > 0,
            "bengali_percentage": (len(bengali_matches) / len(text.split())) * 100 if text.split() else 0
        }
    
    def process_multiple_pdfs(self, pdf_directory: str) -> List[Dict[str, Any]]:
        """Process multiple PDF files from a directory"""
        pdf_path = Path(pdf_directory)
        results = []
        
        for pdf_file in pdf_path.glob("*.pdf"):
            result = self.process_pdf(str(pdf_file))
            results.append(result)
            
        return results

# Example usage and testing
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Test with a sample PDF (you'll need to add your Bengali PDF)
    test_pdf = "data/pdfs/hsc26_bangla_1st_paper.pdf"  # Add your PDF here
    
    if Path(test_pdf).exists():
        result = processor.process_pdf(test_pdf)
        print(f"Processing result: {result['success']}")
        if result['success']:
            print(f"Text length: {len(result['text'])}")
            print(f"Stats: {result['stats']}")
            print(f"First 500 chars: {result['text'][:500]}...")
    else:
        print(f"Please add your Bengali PDF to {test_pdf}")
