"""
Script to download papers from PubMed/PMC for a specific author.
Downloads papers by "Kao CH" and saves them with metadata.
"""

import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
import argparse

try:
    from Bio import Entrez
    from Bio.Entrez import efetch, esearch
except ImportError:
    print("Error: biopython not installed. Run: pip install biopython")
    exit(1)


# Configure Entrez email (required by NCBI)
Entrez.email = "drkalex@umich.edu"  # Change this to your email


def search_pubmed_author(author_name: str, max_results: int = 1000, open_access_only: bool = True) -> List[str]:
    """
    Search PubMed for papers by a specific author.
    
    Args:
        author_name: Author name in format "Last F" or "Last FM"
        max_results: Maximum number of results to return
        open_access_only: If True, only return papers available in PMC (open access)
        
    Returns:
        List of PubMed IDs (PMIDs)
    """
    print(f"Searching PubMed for author: {author_name}")
    if open_access_only:
        print("  Filtering for open access papers (available in PMC)")
    
    # Search query: author name in format [Author]
    # Add filter for free full text (open access) if requested
    search_query = f'"{author_name}"[Author]'
    if open_access_only:
        # Filter for papers available in PMC (open access)
        search_query += " AND free full text[filter]"
    
    try:
        # Search PubMed
        handle = esearch(db="pubmed", term=search_query, retmax=max_results, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        print(f"Found {len(pmids)} papers in PubMed" + (" (open access only)" if open_access_only else ""))
        return pmids
        
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []


def get_pmcid_from_pmid(pmids: List[str], limit: Optional[int] = None) -> Dict[str, str]:
    """
    Convert PubMed IDs to PMC IDs for papers available in PMC.
    
    Args:
        pmids: List of PubMed IDs
        limit: Optional limit on number of PMIDs to process (for testing)
        
    Returns:
        Dictionary mapping PMID -> PMCID
    """
    if not pmids:
        return {}
    
    # Apply limit if specified (for testing)
    if limit:
        pmids = pmids[:limit]
        print(f"TEST MODE: Processing only first {limit} PMIDs")
    
    print(f"Converting {len(pmids)} PMIDs to PMCIDs...")
    pmid_to_pmcid = {}
    
    # Process in batches of 200 (NCBI limit)
    batch_size = 200
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        
        try:
            # Use elink to find PMC IDs
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=batch)
            record = Entrez.read(handle)
            handle.close()
            
            for link_set in record:
                pmid = link_set["IdList"][0]
                if "LinkSetDb" in link_set and len(link_set["LinkSetDb"]) > 0:
                    links = link_set["LinkSetDb"][0]["Link"]
                    if links:
                        pmcid = links[0]["Id"]
                        pmid_to_pmcid[pmid] = pmcid
                        print(f"  Found PMCID: {pmcid} for PMID: {pmid}")
            
            # Be nice to NCBI servers
            time.sleep(0.34)  # ~3 requests per second
            
        except Exception as e:
            print(f"Error converting batch {i//batch_size + 1}: {e}")
            continue
    
    print(f"Found {len(pmid_to_pmcid)} papers available in PMC")
    return pmid_to_pmcid


def download_pmc_article(pmcid: str, output_dir: Path) -> Optional[str]:
    """
    Download full-text article from PubMed Central.
    
    Args:
        pmcid: PMC ID (with or without PMC prefix)
        output_dir: Directory to save the article
        
    Returns:
        Path to saved file if successful, None otherwise
    """
    # Remove PMC prefix if present
    if pmcid.startswith("PMC"):
        pmcid = pmcid[3:]
    
    try:
        # Fetch full-text XML from PMC
        handle = efetch(db="pmc", id=pmcid, rettype="xml", retmode="text")
        xml_content = handle.read()
        handle.close()
        
        # Save XML file
        xml_file = output_dir / f"PMC{pmcid}.xml"
        with open(xml_file, "wb") as f:
            f.write(xml_content)
        
        return str(xml_file)
        
    except Exception as e:
        print(f"Error downloading PMC{pmcid}: {e}")
        return None


def extract_pubmed_metadata(pmid: str) -> Dict:
    """
    Extract metadata for a PubMed article.
    
    Args:
        pmid: PubMed ID
        
    Returns:
        Dictionary with metadata
    """
    try:
        handle = efetch(db="pubmed", id=pmid, rettype="medline", retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        if not record:
            return {}
        
        article = record[0]
        
        metadata = {
            "pmid": pmid,
            "title": article.get("MedlineCitation", {}).get("Article", {}).get("ArticleTitle", ""),
            "authors": [],
            "journal": article.get("MedlineCitation", {}).get("Article", {}).get("Journal", {}).get("Title", ""),
            "year": "",
            "abstract": "",
            "pmcid": None
        }
        
        # Extract authors
        author_list = article.get("MedlineCitation", {}).get("Article", {}).get("AuthorList", [])
        for author in author_list:
            if "LastName" in author and "ForeName" in author:
                metadata["authors"].append(f"{author['LastName']} {author['ForeName']}")
        
        # Extract year
        pub_date = article.get("MedlineCitation", {}).get("Article", {}).get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        if "Year" in pub_date:
            metadata["year"] = pub_date["Year"]
        
        # Extract abstract
        abstract_parts = article.get("MedlineCitation", {}).get("Article", {}).get("Abstract", {}).get("AbstractText", [])
        if isinstance(abstract_parts, list):
            metadata["abstract"] = " ".join([str(part) for part in abstract_parts])
        elif isinstance(abstract_parts, str):
            metadata["abstract"] = abstract_parts
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata for PMID {pmid}: {e}")
        return {"pmid": pmid}


def extract_text_from_pmc_xml(xml_file: str) -> str:
    """
    Extract plain text from PMC XML file.
    
    Args:
        xml_file: Path to PMC XML file
        
    Returns:
        Extracted text content
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Check if publisher restricts full-text download
        xml_content = open(xml_file, 'r', encoding='utf-8', errors='ignore').read()
        if 'does not allow downloading of the full text' in xml_content:
            print(f"  Note: Publisher restricts full-text XML download")
            # Still try to extract abstract and other available content
        
        text_parts = []
        
        # PMC XML uses JATS format without explicit namespace prefixes
        # Extract from abstract first
        for abstract in root.findall(".//abstract"):
            for para in abstract.findall(".//p"):
                para_text = _extract_text_from_element(para)
                if para_text:
                    text_parts.append(para_text)
        
        # Extract from body (may not exist if publisher restricts)
        for body in root.findall(".//body"):
            for sec in body.findall(".//sec"):
                # Extract section title if present
                for title in sec.findall(".//title"):
                    title_text = _extract_text_from_element(title)
                    if title_text:
                        text_parts.append(title_text)
                
                # Extract paragraphs
                for para in sec.findall(".//p"):
                    para_text = _extract_text_from_element(para)
                    if para_text:
                        text_parts.append(para_text)
            
            # Also try direct paragraphs (not in sections)
            for para in body.findall(".//p"):
                para_text = _extract_text_from_element(para)
                if para_text and para_text not in text_parts:
                    text_parts.append(para_text)
        
        # If no body content, extract from abstract and other available sections
        if not text_parts:
            # Try to get any text content
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        print(f"Error extracting text from {xml_file}: {e}")
        return ""


def _extract_text_from_element(elem) -> str:
    """
    Extract all text from an XML element, including text from child elements.
    
    Args:
        elem: XML element
        
    Returns:
        Combined text content
    """
    if elem is None:
        return ""
    
    # Get direct text
    texts = [elem.text] if elem.text else []
    
    # Get text from all children
    for child in elem:
        if child.text:
            texts.append(child.text)
        if child.tail:
            texts.append(child.tail)
    
    # Join and clean up
    combined = " ".join(texts).strip()
    # Clean up extra whitespace
    combined = " ".join(combined.split())
    
    return combined


def main():
    """Main function to download corpus for author."""
    parser = argparse.ArgumentParser(description="Download papers from PubMed/PMC for a specific author")
    parser.add_argument("--author", "-a", default="Kao CH", help="Author name (default: 'Kao CH')")
    parser.add_argument("--limit", "-l", type=int, default=None, 
                       help="Limit number of papers to process (for testing, e.g., -l 5)")
    parser.add_argument("--output", "-o", default="corpus", help="Output directory (default: 'corpus')")
    parser.add_argument("--all-papers", action="store_true", 
                       help="Include all papers (not just open access). Default: open access only")
    parser.add_argument("--min-words", type=int, default=500,
                       help="Minimum word count to consider as full-text (default: 500). Papers below this will be skipped")
    parser.add_argument("--save-abstracts", action="store_true",
                       help="Save abstract-only papers in a separate folder instead of skipping them")
    
    args = parser.parse_args()
    
    author_name = args.author
    output_base = Path(args.output)
    test_limit = args.limit
    open_access_only = not args.all_papers  # Default to open access only
    min_words = args.min_words
    save_abstracts = args.save_abstracts
    
    # Create output directories
    xml_dir = output_base / "xml"
    text_dir = output_base / "text"
    metadata_dir = output_base / "metadata"
    abstracts_dir = output_base / "abstracts_only"  # For papers without full-text
    
    for dir_path in [xml_dir, text_dir, metadata_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if save_abstracts:
        abstracts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting corpus download for author: {author_name}")
    print(f"Output directory: {output_base.absolute()}")
    if open_access_only:
        print("Mode: Open access papers only (will skip papers without PMC access)")
    else:
        print("Mode: All papers (including those without full-text access)")
    print(f"Full-text threshold: Minimum {min_words} words")
    if save_abstracts:
        print("Abstract-only papers will be saved to 'abstracts_only' folder")
    else:
        print("Papers without full-text will be skipped")
    if test_limit:
        print(f"TEST MODE: Limiting to {test_limit} papers")
    print()
    
    # Step 1: Search PubMed
    pmids = search_pubmed_author(author_name, max_results=1000, open_access_only=open_access_only)
    
    if not pmids:
        print("No papers found. Exiting.")
        return
    
    # Apply limit if specified (for testing)
    if test_limit:
        pmids = pmids[:test_limit]
        print(f"\nTEST MODE: Processing only first {test_limit} papers (found {len(pmids)} total)")
    else:
        print(f"\nFound {len(pmids)} papers. Processing...")
    
    # Step 2: Get PMC IDs for papers available in PMC
    pmid_to_pmcid = get_pmcid_from_pmid(pmids, limit=None)  # Already limited above
    
    # Step 3: Filter to only process papers with PMC access (if open_access_only mode)
    if open_access_only:
        pmids_with_access = [pmid for pmid in pmids if pmid in pmid_to_pmcid]
        if len(pmids_with_access) < len(pmids):
            skipped = len(pmids) - len(pmids_with_access)
            print(f"\nFiltering: {skipped} papers without PMC access will be skipped")
            print(f"Processing {len(pmids_with_access)} papers with full-text access")
            pmids = pmids_with_access
    
    # Step 4: Download full-text articles and metadata
    downloaded_count = 0
    skipped_count = 0
    abstract_only_count = 0
    metadata_list = []
    
    print("\nDownloading articles...")
    for i, pmid in enumerate(pmids, 1):
        print(f"\n[{i}/{len(pmids)}] Processing PMID: {pmid}")
        
        # Skip if not available in PMC (shouldn't happen in open_access_only mode)
        if pmid not in pmid_to_pmcid:
            print(f"  Skipping: No PMC access available")
            skipped_count += 1
            continue
        
        # Get metadata
        metadata = extract_pubmed_metadata(pmid)
        
        # Download full-text from PMC
        pmcid = pmid_to_pmcid[pmid]
        metadata["pmcid"] = pmcid
        
        print(f"  Downloading full-text from PMC: {pmcid}")
        xml_file = download_pmc_article(pmcid, xml_dir)
        
        if xml_file:
            # Extract text from XML
            text_content = extract_text_from_pmc_xml(xml_file)
            
            if text_content:
                word_count = len(text_content.split())
                metadata["word_count"] = word_count
                
                # Check if we have full-text or just abstract
                if word_count < min_words:
                    # Abstract only or very short content
                    if save_abstracts:
                        # Save to abstracts folder
                        abstract_file = abstracts_dir / f"PMC{pmcid}.txt"
                        with open(abstract_file, "w", encoding="utf-8") as f:
                            f.write(text_content)
                        metadata["text_file"] = str(abstract_file)
                        metadata["abstract_only"] = True
                        abstract_only_count += 1
                        print(f"  Saved abstract-only: {abstract_file} ({word_count} words) - below threshold")
                    else:
                        # Skip it
                        print(f"  Skipping: Only abstract/extracted content ({word_count} words) - below {min_words} word threshold")
                        skipped_count += 1
                        # Don't add to metadata_list, just continue
                        continue
                else:
                    # Full-text article
                    text_file = text_dir / f"PMC{pmcid}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(text_content)
                    metadata["text_file"] = str(text_file)
                    metadata["abstract_only"] = False
                    downloaded_count += 1
                    print(f"  Saved full-text: {text_file} ({word_count} words)")
            else:
                print(f"  Skipping: No text content extracted from XML")
                skipped_count += 1
                continue
        else:
            print(f"  Skipping: Failed to download XML file")
            skipped_count += 1
            continue
        
        # Save metadata (only if we kept the paper)
        metadata_file = metadata_dir / f"PMID_{pmid}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        metadata_list.append(metadata)
        
        # Brief pause between requests
        time.sleep(0.34)
    
    # Step 5: Save summary
    summary = {
        "author": author_name,
        "open_access_only": open_access_only,
        "min_words_threshold": min_words,
        "save_abstracts": save_abstracts,
        "total_papers": len(pmids),
        "pmc_available": len(pmid_to_pmcid),
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
    print(f"Total papers found: {len(pmids)}")
    print(f"Papers available in PMC: {len(pmid_to_pmcid)}")
    print(f"Full-text articles downloaded: {downloaded_count}")
    if save_abstracts:
        print(f"Abstract-only papers saved: {abstract_only_count}")
    print(f"Papers skipped: {skipped_count}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()