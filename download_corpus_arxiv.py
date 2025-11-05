"""
Script to download papers from arXiv for a specific author.
Downloads papers by a specified author and saves them with metadata.
"""

import os
import time
import re
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


def normalize_author_name(name: str) -> str:
    """
    Normalize author name for comparison (case-insensitive, handle variations).
    
    Args:
        name: Author name in various formats
        
    Returns:
        Normalized name for comparison
    """
    # Remove extra spaces, convert to lowercase
    normalized = " ".join(name.split()).lower()
    # Handle "Last, First" vs "Last First" formats
    if "," in normalized:
        parts = [p.strip() for p in normalized.split(",")]
        normalized = " ".join(reversed(parts))  # Convert to "First Last"
    return normalized


def author_matches(author_list: List[str], target_author: str) -> bool:
    """
    Check if any author in the list matches the target author.
    
    Args:
        author_list: List of author names from paper
        target_author: Target author name to match
        
    Returns:
        True if there's a match, False otherwise
    """
    target_normalized = normalize_author_name(target_author)
    
    for author in author_list:
        author_normalized = normalize_author_name(author)
        # Check if names match (allowing for middle initial variations)
        if target_normalized == author_normalized:
            return True
        # Also check if one is a substring of the other (handles "Last F" vs "Last First")
        if target_normalized in author_normalized or author_normalized in target_normalized:
            # Make sure it's not just a partial match (e.g., "Smith" matching "Smithson")
            words_target = set(target_normalized.split())
            words_author = set(author_normalized.split())
            if len(words_target) > 0 and words_target.issubset(words_author):
                return True
            if len(words_author) > 0 and words_author.issubset(words_target):
                return True
    
    return False


def get_paper_versions(arxiv_id: str) -> Dict[str, int]:
    """
    Get version information for an arXiv paper.
    
    Args:
        arxiv_id: arXiv ID without version suffix (e.g., "1234.5678")
        
    Returns:
        Dictionary with 'latest_version' and 'has_v1' keys
    """
    try:
        # Query arXiv API to get version info
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "id_list": arxiv_id,
            "max_results": 1
        }
        
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        feed = feedparser.parse(url)
        
        if feed.entries:
            entry = feed.entries[0]
            # Extract version from ID (format: http://arxiv.org/abs/1234.5678v3)
            entry_id = entry.id.split('/')[-1]
            if 'v' in entry_id:
                version_str = entry_id.split('v')[1]
                latest_version = int(version_str)
            else:
                latest_version = 1
            
            return {
                "latest_version": latest_version,
                "has_v1": latest_version >= 1
            }
    except Exception as e:
        print(f"  Warning: Could not get version info for {arxiv_id}: {e}")
    
    # Default: assume v1 exists and is latest
    return {"latest_version": 1, "has_v1": True}


def search_arxiv_author(author_name: str, max_results: int = 1000) -> List[Dict]:
    """
    Search arXiv for papers by a specific author, filtering to ensure exact author match.
    
    Args:
        author_name: Author name (can be "Last F" or "Last, First")
        max_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries with arXiv IDs and metadata (only papers by exact author)
    """
    print(f"Searching arXiv for author: {author_name}")
    
    # arXiv search: au = author search
    # Try different formats for author name
    search_queries = [
        f'au:"{author_name}"',
        f'au:"{author_name.replace(" ", ", ")}"',  # "Last F" -> "Last, F"
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
                    # Get authors from entry
                    authors = [author.name for author in entry.authors]
                    
                    # Filter: only include papers where the target author is actually in the author list
                    if author_matches(authors, author_name):
                        seen_ids.add(arxiv_id)
                        all_papers.append({
                            "arxiv_id": arxiv_id,
                            "title": entry.title,
                            "authors": authors,
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
    
    print(f"Found {len(all_papers)} papers by {author_name} in arXiv")
    return all_papers


def download_arxiv_pdf(arxiv_id: str, version: Optional[int] = None, output_dir: Path = None) -> Optional[str]:
    """
    Download PDF from arXiv for a specific version.
    
    Args:
        arxiv_id: arXiv ID (without version suffix)
        version: Version number (1, 2, etc.). If None, downloads latest version
        output_dir: Directory to save the PDF
        
    Returns:
        Path to saved PDF file if successful, None otherwise
    """
    try:
        # Construct PDF URL with version
        if version:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{version}.pdf"
            pdf_file = output_dir / f"{arxiv_id}_v{version}.pdf"
        else:
            # Latest version (no version suffix defaults to latest)
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_file = output_dir / f"{arxiv_id}_latest.pdf"
        
        # Download PDF
        urllib.request.urlretrieve(pdf_url, pdf_file)
        
        return str(pdf_file)
        
    except Exception as e:
        print(f"Error downloading PDF for {arxiv_id}v{version if version else 'latest'}: {e}")
        return None


def extract_text_from_pdf(pdf_file: str) -> str:
    """
    Extract plain text from PDF file.
    Tries pdfplumber first (better quality), falls back to PyPDF2.
    
    Args:
        pdf_file: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        text_parts = []
        
        # Try pdfplumber first (better text extraction quality)
        try:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Clean up common PDF extraction artifacts
                        text = text.replace('\x00', '')  # Remove null bytes
                        text = text.replace('\r\n', '\n')  # Normalize line endings
                        text_parts.append(text.strip())
            
            if text_parts:
                combined = "\n\n".join(text_parts)
                # Remove excessive whitespace
                import re
                combined = re.sub(r'\n{3,}', '\n\n', combined)  # Max 2 newlines
                return combined.strip()
        
        except ImportError:
            pass  # Fall back to PyPDF2
        except Exception as e:
            print(f"  Warning: pdfplumber extraction failed, trying PyPDF2: {e}")
        
        # Fallback to PyPDF2
        with open(pdf_file, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    # Clean up common PDF extraction artifacts
                    text = text.replace('\x00', '')  # Remove null bytes
                    text = text.replace('\r\n', '\n')  # Normalize line endings
                    text_parts.append(text.strip())
        
        if text_parts:
            combined = "\n\n".join(text_parts)
            # Remove excessive whitespace
            import re
            combined = re.sub(r'\n{3,}', '\n\n', combined)  # Max 2 newlines
            return combined.strip()
        
        return ""
        
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
    print("Note: Downloading both v1 (first version) and latest version for each paper")
    
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper["arxiv_id"]
        print(f"\n[{i}/{len(papers)}] Processing arXiv ID: {arxiv_id}")
        print(f"  Title: {paper.get('title', 'N/A')[:80]}...")
        
        # Get version information
        version_info = get_paper_versions(arxiv_id)
        latest_version = version_info["latest_version"]
        has_v1 = version_info["has_v1"]
        
        print(f"  Versions available: v1 through v{latest_version}")
        
        # Get base metadata
        base_metadata = extract_metadata(paper)
        base_metadata["versions"] = {
            "latest_version": latest_version,
            "has_v1": has_v1
        }
        
        versions_downloaded = []
        
        # Download v1 (first version) if available
        if has_v1:
            print(f"  Downloading v1 (first version)...")
            pdf_file_v1 = download_arxiv_pdf(arxiv_id, version=1, output_dir=pdf_dir)
            
            if pdf_file_v1:
                text_content_v1 = extract_text_from_pdf(pdf_file_v1)
                
                if text_content_v1:
                    word_count_v1 = len(text_content_v1.split())
                    
                    if word_count_v1 >= min_words:
                        text_file_v1 = text_dir / f"{arxiv_id}_v1.txt"
                        with open(text_file_v1, "w", encoding="utf-8") as f:
                            f.write(text_content_v1)
                        
                        versions_downloaded.append({
                            "version": 1,
                            "text_file": str(text_file_v1),
                            "pdf_file": str(pdf_file_v1),
                            "word_count": word_count_v1
                        })
                        print(f"    Saved v1: {text_file_v1} ({word_count_v1} words)")
                    else:
                        print(f"    v1 skipped: Only {word_count_v1} words (below threshold)")
                else:
                    print(f"    v1 skipped: No text extracted")
            else:
                print(f"    v1 skipped: Download failed")
        
        # Download latest version (if different from v1)
        if latest_version > 1:
            print(f"  Downloading v{latest_version} (latest version)...")
            pdf_file_latest = download_arxiv_pdf(arxiv_id, version=latest_version, output_dir=pdf_dir)
            
            if pdf_file_latest:
                text_content_latest = extract_text_from_pdf(pdf_file_latest)
                
                if text_content_latest:
                    word_count_latest = len(text_content_latest.split())
                    
                    if word_count_latest >= min_words:
                        text_file_latest = text_dir / f"{arxiv_id}_v{latest_version}.txt"
                        with open(text_file_latest, "w", encoding="utf-8") as f:
                            f.write(text_content_latest)
                        
                        versions_downloaded.append({
                            "version": latest_version,
                            "text_file": str(text_file_latest),
                            "pdf_file": str(pdf_file_latest),
                            "word_count": word_count_latest
                        })
                        print(f"    Saved v{latest_version}: {text_file_latest} ({word_count_latest} words)")
                    else:
                        print(f"    v{latest_version} skipped: Only {word_count_latest} words (below threshold)")
                else:
                    print(f"    v{latest_version} skipped: No text extracted")
            else:
                print(f"    v{latest_version} skipped: Download failed")
        elif latest_version == 1 and len(versions_downloaded) == 0:
            # If only v1 exists and we didn't download it above, try again
            print(f"  Downloading latest version (v1)...")
            pdf_file = download_arxiv_pdf(arxiv_id, version=None, output_dir=pdf_dir)
            
            if pdf_file:
                text_content = extract_text_from_pdf(pdf_file)
                
                if text_content:
                    word_count = len(text_content.split())
                    
                    if word_count >= min_words:
                        text_file = text_dir / f"{arxiv_id}_v1.txt"
                        with open(text_file, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        
                        versions_downloaded.append({
                            "version": 1,
                            "text_file": str(text_file),
                            "pdf_file": str(pdf_file),
                            "word_count": word_count
                        })
                        print(f"    Saved v1: {text_file} ({word_count} words)")
        
        # Update counts and metadata
        if versions_downloaded:
            base_metadata["versions_downloaded"] = versions_downloaded
            base_metadata["total_versions"] = len(versions_downloaded)
            downloaded_count += len(versions_downloaded)
            
            # Save metadata
            metadata_file = metadata_dir / f"{arxiv_id}.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(base_metadata, f, indent=2, ensure_ascii=False)
            
            metadata_list.append(base_metadata)
        else:
            print(f"  Skipping paper: No versions met word threshold")
            skipped_count += 1
        
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

