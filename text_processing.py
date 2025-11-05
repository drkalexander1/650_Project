"""
Text processing utilities for corpus extraction.
Contains functions for cleaning and processing extracted text from PDFs.
"""

import re
from typing import Optional


def remove_references_section(text: str) -> str:
    """
    Remove reference sections from extracted text using regex patterns.
    
    Args:
        text: Full text content from PDF
        
    Returns:
        Text with reference section removed
    """
    if not text:
        return text
    
    # Patterns for reference section headers (case-insensitive)
    reference_patterns = [
        r'\n\s*(?:References|Bibliography|Works Cited|References and Notes|Literature Cited|Citations|Cited References)',
        r'\n\s*(?:REFERENCES|BIBLIOGRAPHY|WORKS CITED|LITERATURE CITED)',
        r'\n\s*(?:References\s*$|Bibliography\s*$)',
        r'\n\s*\[\s*\d+\s*\]\s*[A-Z]',  # Pattern like "[1] Author" at start of references
        r'\n\s*\d+\.\s*[A-Z][a-z]+\s+[A-Z]',  # Pattern like "1. Author Name" at start
    ]
    
    # Try to find where references section starts
    reference_start = None
    
    for pattern in reference_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            # Take the last match (most likely to be the actual references section)
            # but also check if it's in the last 30% of the document
            for match in reversed(matches):
                position = match.start()
                # References usually appear in the last 30% of the document
                if position > len(text) * 0.7:
                    reference_start = position
                    break
            if reference_start:
                break
    
    # If we found a reference section start, remove everything from there
    if reference_start:
        # Also check for common patterns that indicate references section
        # Look for citation-like patterns near the match
        section_text = text[reference_start:reference_start+500]
        
        # Count citation indicators
        citation_indicators = [
            r'\(\d{4}\)',  # Years in parentheses like (2020)
            r'\[\d+\]',    # Numbered citations like [1]
            r'\d+\.\s*[A-Z]',  # Numbered list with capital letter
            r'[A-Z][a-z]+\s+et\s+al\.',  # "Author et al."
            r'https?://',  # URLs (common in references)
            r'doi:',       # DOI references
            r'arXiv:',     # arXiv references
        ]
        
        indicator_count = sum(len(re.findall(pattern, section_text, re.IGNORECASE)) 
                            for pattern in citation_indicators)
        
        # If we have multiple citation indicators, it's likely references
        if indicator_count >= 3:
            removed_text = text[:reference_start].rstrip()
            # Remove any trailing section headers that might be incomplete
            removed_text = re.sub(r'\n\s*(?:References?|Bibliography|Works Cited).*$', '', 
                                 removed_text, flags=re.IGNORECASE)
            return removed_text
    
    # Fallback: Try to find common patterns that indicate end of main content
    # Look for "References" or "Bibliography" anywhere and remove from there
    for pattern in [r'\n\s*References\s*\n', r'\n\s*Bibliography\s*\n']:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            position = match.start()
            # Only remove if it's in the last 40% of document
            if position > len(text) * 0.6:
                return text[:position].rstrip()
    
    return text


def clean_extracted_text(text: str, remove_references: bool = True) -> str:
    """
    Clean extracted text from PDFs.
    
    Args:
        text: Raw extracted text
        remove_references: If True, remove reference sections
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    
    # Remove references if requested
    if remove_references:
        cleaned = remove_references_section(cleaned)
    
    return cleaned.strip()


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in extracted text.
    
    Args:
        text: Text with potentially inconsistent whitespace
        
    Returns:
        Text with normalized whitespace
    """
    if not text:
        return text
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    
    return text.strip()

