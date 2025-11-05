"""
Script to download papers from arXiv for a specific author.
Downloads papers by a specified author and saves them with metadata.
"""

import os
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
import argparse
import xml.etree.ElementTree as ET

try:
    import feedparser
except ImportError:
    print("Error: feedparser not installed. Run: pip install feedparser")
    exit(1)

try:
    import PyPDF2
except ImportError:
    try:
        import pdfplumber
        PDF_EXTRACTOR = "pdfplumber"
    except ImportError:
        print("Error: Need either PyPDF2 or pdfplumber for PDF extraction.")
        print("Run: pip install PyPDF2 or pip install pdfplumber")
        exit(1)
else:
    PDF_EXTRACTOR = "PyPDF2"


def search_arxiv_author(author_name: str, max_results: int = 1000) -> List[Dict]:
    """
    Search arXiv for papers by a specific author.
    
    Args:
        author_name: Author name (can be "Last F" or "Last, First")
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with arXiv IDs and metadata
    """
    print(f"Searching arXiv for author: {author_name}")
    
    # arXiv search: au = author search
    # Try different formats for author name
    search_queries = [
        f'au:"{author_name}"',
        f'au:"{author_name.replace(" ", ", ")}"',  # "Last F" -> "Last, F"
        f'au:"{author_name.replace(" ", " ")}"',   # Keep as is
    ]
    
    all_papers = []
    seen_ids = set()
    
    for query in search_queries:
        try:
            # arXiv API endpoint
            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                # Extract arXiv ID (format: http://arxiv.org/abs/1234.5678v1 -> 1234.5678)
                arxiv_id = entry.id.split('/')[-1].split('v')[0]  # Remove version suffix
                
                if arxiv_id not in seen_ids:
                    seen_ids.add(arxiv_id)
                    all_papers.append({
                        "arxiv_id": arxiv_id,
                        "title": entry.title,
                        "authors": [author.name for author in entry.authors],
                        "summary": entry.summary,
                        "published": entry.published,
                        "pdf_url": None  # Will be set below
                    })
                    
                    # Find PDF URL
                    for link in entry.links:
                        if link.rel == "alternate" and "pdf" in link.type:
                            all_papers[-1]["pdf_url"] = link.href
                        elif link.type == "application/pdf":
                            all_papers[-1]["pdf_url"] = link.href
            
            # Be nice to arXiv servers
            time.sleep(1)
            
            if all_papers:
                break  # If we found papers, use this query
                
        except Exception as e:
            print(f"Error with query '{query}': {e}")
            continue
    
    print(f"Found {len(all_papers)} papers in arXiv")
    return all_papers


def download_arxiv_pdf(arxiv_id: str, pdf_url: str, output_dir: Path) -> Optional[str]:
    """
    Download PDF from arXiv.
    
    Args:
        arxiv_id: arXiv ID
        pdf_url: URL to PDF
        output_dir: Directory to save the PDF
        
    Returns:
        Path to saved PDF file if successful, None otherwise
    """
    try:
        if not pdf_url:
            # Construct PDF URL from arXiv ID
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        pdf_file = output_dir / f"{arxiv_id}.pdf"
        
        # Download PDF
        urllib.request.urlretrieve(pdf_url, pdf_file)
        
        return str(pdf_file)
        
    except Exception as e:
        print(f"Error downloading PDF for {arxiv_id}: {e}")
        return None


def extract_text_from_pdf(pdf_file: str) -> str:
    """
    Extract plain text from PDF file.
    
    Args:
        pdf_file: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        text_parts = []
        
        if PDF_EXTRACTOR == "PyPDF2":
            with open(pdf_file, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
        
        elif PDF_EXTRACTOR == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
        return ""


def extract_metadata(paper: Dict) -> Dict:
    """
    Extract metadata from arXiv paper entry.
    
    Args:
        paper: Dictionary with paper information from arXiv API
        
    Returns:
        Dictionary with metadata
    """
    metadata = {
        "arxiv_id": paper["arxiv_id"],
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("summary", ""),
        "published": paper.get("published", ""),
        "year": "",
        "pdf_url": paper.get("pdf_url", "")
    }
    
    # Extract year from published date
    try:
        if paper.get("published"):
            year = paper["published"].split("-")[0]
            metadata["year"] = year
    except:
        pass
    
    return metadata


def main():
    """Main function to download corpus for author."""
    parser = argparse.ArgumentParser(description="Download papers from arXiv for a specific author")
    parser.add_argument("--author", "-a", default="Kao CH", help="Author name (default: 'Kao CH')")
    parser.add_argument("--limit", "-l", type=int, default=None, 
                       help="Limit number of papers to process (for testing, e.g., -l 5)")
    parser.add_argument("--output", "-o", default="corpus", help="Output directory (default: 'corpus')")
    parser.add_argument("--min-words", type=int, default=500,
                       help="Minimum word count to consider as full-text (default: 500). Papers below this will be skipped")
    parser.add_argument("--save-abstracts", action="store_true",
                       help="Save abstract-only papers in a separate folder instead of skipping them")
    
    args = parser.parse_args()
    
    author_name = args.author
    output_base = Path(args.output)
    test_limit = args.limit
    min_words = args.min_words
    save_abstracts = args.save_abstracts
    
    # Create output directories
    pdf_dir = output_base / "pdf"
    text_dir = output_base / "text"
    metadata_dir = output_base / "metadata"
    abstracts_dir = output_base / "abstracts_only"  # For papers without full-text
    
    for dir_path in [pdf_dir, text_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if save_abstracts:
        abstracts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting corpus download for author: {author_name}")
    print(f"Output directory: {output_base.absolute()}")
    print(f"Source: arXiv")
    print(f"Full-text threshold: Minimum {min_words} words")
    if save_abstracts:
        print("Abstract-only papers will be saved to 'abstracts_only' folder")
    else:
        print("Papers without full-text will be skipped")
    if test_limit:
        print(f"TEST MODE: Limiting to {test_limit} papers")
    print()
    
    # Step 1: Search arXiv
    papers = search_arxiv_author(author_name, max_results=1000)
    
    if not papers:
        print("No papers found. Exiting.")
        return
    
    # Apply limit if specified (for testing)
    if test_limit:
        papers = papers[:test_limit]
        print(f"\nTEST MODE: Processing only first {test_limit} papers (found {len(papers)} total)")
    else:
        print(f"\nFound {len(papers)} papers. Processing...")
    
    # Step 2: Download papers and extract text
    downloaded_count = 0
    skipped_count = 0
    abstract_only_count = 0
    metadata_list = []
    
    print("\nDownloading articles...")
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["arxiv_id"]
        print(f"\n[{i}/{len(papers)}] Processing arXiv ID: {arxiv_id}")
        print(f"  Title: {paper.get('title', 'N/A')[:80]}...")
        
        # Get metadata
        metadata = extract_metadata(paper)
        
        # Download PDF
        pdf_url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"  Downloading PDF from arXiv...")
        pdf_file = download_arxiv_pdf(arxiv_id, pdf_url, pdf_dir)
        
        if pdf_file:
            # Extract text from PDF
            text_content = extract_text_from_pdf(pdf_file)
            
            if text_content:
                word_count = len(text_content.split())
                metadata["word_count"] = word_count
                
                # Check if we have full-text or just abstract
                if word_count < min_words:
                    # Abstract only or very short content
                    if save_abstracts:
                        # Save to abstracts folder
                        abstract_file = abstracts_dir / f"{arxiv_id}.txt"
                        with open(abstract_file, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        metadata["text_file"] = str(abstract_file)
                        metadata["abstract_only"] = True
                        abstract_only_count += 1
                        print(f"  Saved abstract-only: {abstract_file} ({word_count} words) - below threshold")
                    else:
                        # Skip it
                        print(f"  Skipping: Only {word_count} words - below {min_words} word threshold")
                        skipped_count += 1
                        continue
                else:
                    # Full-text article
                    text_file = text_dir / f"{arxiv_id}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    metadata["text_file"] = str(text_file)
                    metadata["pdf_file"] = str(pdf_file)
                    metadata["abstract_only"] = False
                    downloaded_count += 1
                    print(f"  Saved full-text: {text_file} ({word_count} words)")
            else:
                print(f"  Skipping: No text content extracted from PDF")
                skipped_count += 1
                continue
        else:
            print(f"  Skipping: Failed to download PDF")
            skipped_count += 1
            continue
        
        # Save metadata (only if we kept the paper)
        metadata_file = metadata_dir / f"{arxiv_id}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        metadata_list.append(metadata)
        
        # Brief pause between requests
        time.sleep(1)
    
    # Step 3: Save summary
    summary = {
        "author": author_name,
        "source": "arXiv",
        "min_words_threshold": min_words,
        "save_abstracts": save_abstracts,
        "total_papers": len(papers),
        "full_text_downloaded": downloaded_count,
        "abstract_only_saved": abstract_only_count if save_abstracts else 0,
        "skipped": skipped_count,
        "download_date": datetime.now().isoformat(),
        "test_mode": test_limit is not None,
        "limit": test_limit,
        "papers": metadata_list
    }
    
    summary_file = output_base / "corpus_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Total papers found: {len(papers)}")
    print(f"Full-text articles downloaded: {downloaded_count}")
    if save_abstracts:
        print(f"Abstract-only papers saved: {abstract_only_count}")
    print(f"Papers skipped: {skipped_count}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

