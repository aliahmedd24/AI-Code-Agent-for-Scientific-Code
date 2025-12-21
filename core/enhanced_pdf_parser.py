"""
Enhanced PDF Parser - Multi-Backend Text Extraction with Fallback

This module provides robust PDF text extraction by:
- Using multiple extraction backends (PyMuPDF, pdfplumber, OCR)
- Automatically selecting the best result based on quality metrics
- Using LLM to structure poorly extracted content
- Handling various PDF formats and edge cases

Author: Scientific Agent System
"""

import os
import re
import asyncio
import tempfile
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from a PDF extraction attempt."""
    text: str
    confidence: float
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        return self.error is None and len(self.text.strip()) > 100


class PDFExtractorBackend(ABC):
    """Abstract base class for PDF extraction backends."""
    
    name: str = "base"
    
    @abstractmethod
    async def extract(self, pdf_path: str) -> ExtractionResult:
        """Extract text from PDF file."""
        pass
    
    def calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for extracted text."""
        if not text or len(text.strip()) < 50:
            return 0.0
        
        score = 0.0
        
        # Length score (longer is generally better, up to a point)
        length = len(text)
        if length > 1000:
            score += 0.2
        if length > 5000:
            score += 0.1
        if length > 10000:
            score += 0.1
        
        # Word ratio (real text has proper word boundaries)
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if 3 <= avg_word_len <= 10:
                score += 0.2
        
        # Sentence structure (real text has periods, proper sentences)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 10:
            score += 0.1
        
        # Low garbage ratio (special characters, non-printable)
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        if len(text) > 0:
            printable_ratio = printable / len(text)
            if printable_ratio > 0.95:
                score += 0.2
            elif printable_ratio > 0.90:
                score += 0.1
        
        # Has common section headers
        section_patterns = [
            r'\babstract\b', r'\bintroduction\b', r'\bmethod', 
            r'\bresult', r'\bconclusion', r'\breference'
        ]
        section_matches = sum(1 for p in section_patterns if re.search(p, text.lower()))
        score += min(section_matches * 0.05, 0.2)
        
        return min(score, 1.0)


class PyMuPDFExtractor(PDFExtractorBackend):
    """Extract text using PyMuPDF (fitz)."""
    
    name = "pymupdf"
    
    async def extract(self, pdf_path: str) -> ExtractionResult:
        try:
            import fitz
        except ImportError:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error="PyMuPDF (fitz) not installed"
            )
        
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            metadata = {}
            
            # Extract metadata
            if doc.metadata:
                metadata = {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'creator': doc.metadata.get('creator', ''),
                    'page_count': doc.page_count
                }
            
            # Extract text from each page
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            doc.close()
            
            full_text = "\n\n".join(text_parts)
            confidence = self.calculate_quality_score(full_text)
            
            return ExtractionResult(
                text=full_text,
                confidence=confidence,
                backend=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error=str(e)
            )


class PDFPlumberExtractor(PDFExtractorBackend):
    """Extract text using pdfplumber (better for tables and structured content)."""
    
    name = "pdfplumber"
    
    async def extract(self, pdf_path: str) -> ExtractionResult:
        try:
            import pdfplumber
        except ImportError:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error="pdfplumber not installed. Install with: pip install pdfplumber"
            )
        
        try:
            text_parts = []
            tables = []
            metadata = {}
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata['page_count'] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            tables.append({
                                'page': page_num + 1,
                                'data': table
                            })
            
            full_text = "\n\n".join(text_parts)
            
            # Add table data to text
            if tables:
                full_text += "\n\n--- Extracted Tables ---\n"
                for i, table in enumerate(tables):
                    full_text += f"\nTable {i+1} (Page {table['page']}):\n"
                    for row in table['data']:
                        if row:
                            full_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
            
            confidence = self.calculate_quality_score(full_text)
            metadata['tables_found'] = len(tables)
            
            return ExtractionResult(
                text=full_text,
                confidence=confidence,
                backend=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error=str(e)
            )


class OCRExtractor(PDFExtractorBackend):
    """Extract text using OCR (pdf2image + pytesseract) for scanned PDFs."""
    
    name = "ocr"
    
    async def extract(self, pdf_path: str) -> ExtractionResult:
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error="OCR dependencies not installed. Install with: pip install pdf2image pytesseract"
            )
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            text_parts = []
            for i, image in enumerate(images):
                # Run OCR on each page
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            confidence = self.calculate_quality_score(full_text)
            
            # OCR typically has lower baseline confidence
            confidence *= 0.85
            
            return ExtractionResult(
                text=full_text,
                confidence=confidence,
                backend=self.name,
                metadata={'page_count': len(images), 'method': 'OCR'}
            )
            
        except Exception as e:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error=str(e)
            )


class PyPDF2Extractor(PDFExtractorBackend):
    """Extract text using PyPDF2 as another fallback."""
    
    name = "pypdf2"
    
    async def extract(self, pdf_path: str) -> ExtractionResult:
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                return ExtractionResult(
                    text="",
                    confidence=0.0,
                    backend=self.name,
                    error="PyPDF2/pypdf not installed. Install with: pip install pypdf"
                )
        
        try:
            reader = PdfReader(pdf_path)
            text_parts = []
            
            metadata = {}
            if reader.metadata:
                metadata = {
                    'title': reader.metadata.get('/Title', ''),
                    'author': reader.metadata.get('/Author', ''),
                    'page_count': len(reader.pages)
                }
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
            
            full_text = "\n\n".join(text_parts)
            confidence = self.calculate_quality_score(full_text)
            
            return ExtractionResult(
                text=full_text,
                confidence=confidence,
                backend=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend=self.name,
                error=str(e)
            )


class EnhancedPDFParser:
    """
    Enhanced PDF parser with multiple extraction backends and LLM-based structuring.
    
    Features:
    - Tries multiple PDF extraction backends in order of preference
    - Selects the best result based on quality metrics
    - Uses LLM to structure messy extracted content
    - Handles various PDF formats including scanned documents
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the enhanced PDF parser.
        
        Args:
            llm_client: Optional LLM client for content structuring
        """
        self.llm_client = llm_client
        
        # Initialize backends in order of preference
        self.backends: List[PDFExtractorBackend] = [
            PyMuPDFExtractor(),
            PDFPlumberExtractor(),
            PyPDF2Extractor(),
            OCRExtractor(),  # Last resort - slow but handles scanned PDFs
        ]
    
    async def extract_text(self, pdf_path: str) -> ExtractionResult:
        """
        Extract text from PDF using the best available backend.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractionResult with the best extraction
        """
        if not os.path.exists(pdf_path):
            return ExtractionResult(
                text="",
                confidence=0.0,
                backend="none",
                error=f"PDF file not found: {pdf_path}"
            )
        
        results: List[ExtractionResult] = []
        
        # Try each backend
        for backend in self.backends:
            logger.info(f"Trying PDF extraction with {backend.name}...")
            result = await backend.extract(pdf_path)
            results.append(result)
            
            # If we get a high-confidence result, use it immediately
            if result.confidence >= 0.8:
                logger.info(f"High confidence result from {backend.name}: {result.confidence:.2f}")
                break
            
            if result.error:
                logger.warning(f"{backend.name} failed: {result.error}")
        
        # Select the best result
        successful_results = [r for r in results if r.is_successful]
        
        if not successful_results:
            # All backends failed, return the one with most text
            best = max(results, key=lambda r: len(r.text))
            return ExtractionResult(
                text=best.text,
                confidence=0.1,
                backend=best.backend,
                error="All extraction backends produced low-quality results"
            )
        
        # Return the highest confidence result
        best = max(successful_results, key=lambda r: r.confidence)
        logger.info(f"Best extraction from {best.backend} with confidence {best.confidence:.2f}")
        
        return best
    
    async def extract_and_structure(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text and use LLM to structure it into sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Structured paper content
        """
        # First, extract raw text
        extraction = await self.extract_text(pdf_path)
        
        if not extraction.is_successful:
            return {
                'success': False,
                'error': extraction.error,
                'raw_text': extraction.text,
                'backend_used': extraction.backend
            }
        
        # If we have an LLM client, use it to structure the content
        if self.llm_client and extraction.confidence < 0.9:
            structured = await self._structure_with_llm(extraction.text)
            if structured:
                return {
                    'success': True,
                    'raw_text': extraction.text,
                    'structured': structured,
                    'backend_used': extraction.backend,
                    'confidence': extraction.confidence,
                    'llm_structured': True
                }
        
        # Otherwise, try regex-based structuring
        structured = self._structure_with_regex(extraction.text)
        
        return {
            'success': True,
            'raw_text': extraction.text,
            'structured': structured,
            'backend_used': extraction.backend,
            'confidence': extraction.confidence,
            'llm_structured': False
        }
    
    async def _structure_with_llm(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Use LLM to identify and structure paper sections."""
        if not self.llm_client:
            return None
        
        # Truncate if too long
        text_sample = raw_text[:15000] if len(raw_text) > 15000 else raw_text
        
        prompt = f"""Analyze this raw text extracted from a scientific paper PDF and structure it.

The text may have extraction artifacts, strange formatting, or missing content.
Do your best to identify the actual content.

Raw text:
{text_sample}

Identify and return as JSON:
{{
    "title": "The paper title (exact text if found)",
    "authors": ["List of authors if found"],
    "abstract": "The abstract text",
    "sections": [
        {{"title": "Section title", "content": "Section content summary"}},
        ...
    ],
    "keywords": ["key", "terms", "from", "paper"],
    "equations": ["Any equations found in LaTeX format"],
    "extraction_quality": "good|fair|poor",
    "notes": "Any notes about extraction issues"
}}
"""
        
        try:
            response = await self.llm_client.generate_structured(
                prompt,
                schema={"type": "object"}
            )
            return response
        except Exception as e:
            logger.error(f"LLM structuring failed: {e}")
            return None
    
    def _structure_with_regex(self, text: str) -> Dict[str, Any]:
        """Use regex patterns to structure the text."""
        result = {
            'title': '',
            'authors': [],
            'abstract': '',
            'sections': [],
            'keywords': []
        }
        
        lines = text.split('\n')
        
        # Try to find title (usually first non-empty line or after page marker)
        for line in lines[:20]:
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('---'):
                # Skip common non-title patterns
                if not re.match(r'^(page|arxiv|doi|http|www\.)', line.lower()):
                    result['title'] = line
                    break
        
        # Find abstract
        abstract_match = re.search(
            r'(?:abstract|summary)[:\s]*\n?(.*?)(?:\n\s*\n|\n(?:1\.?\s+)?(?:introduction|keywords|$))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            result['abstract'] = abstract_match.group(1).strip()[:2000]
        
        # Find sections
        section_pattern = r'^(?:(\d+\.?\s+)|([IVX]+\.?\s+))?([A-Z][A-Za-z\s]+)$'
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if re.match(section_pattern, line_stripped) and len(line_stripped) < 100:
                if current_section:
                    result['sections'].append({
                        'title': current_section,
                        'content': '\n'.join(current_content)[:3000]
                    })
                current_section = line_stripped
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add last section
        if current_section:
            result['sections'].append({
                'title': current_section,
                'content': '\n'.join(current_content)[:3000]
            })
        
        # Extract keywords
        keywords_match = re.search(
            r'keywords?[:\s]*(.*?)(?:\n\s*\n|$)',
            text,
            re.IGNORECASE
        )
        if keywords_match:
            keywords_text = keywords_match.group(1)
            result['keywords'] = [
                k.strip() for k in re.split(r'[,;•·]', keywords_text)
                if k.strip() and len(k.strip()) < 50
            ]
        
        return result


def create_enhanced_parser(llm_client=None) -> EnhancedPDFParser:
    """Factory function to create an enhanced PDF parser."""
    return EnhancedPDFParser(llm_client=llm_client)
