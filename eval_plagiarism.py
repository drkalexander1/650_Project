"""
Plagiarism Detection System for PAN Plagiarism Corpus 2011
This script detects plagiarism between source documents and suspicious documents.
Source documents are in pan-plagiarism-corpus-2011/external-detection-corpus/source-document/
Suspicious documents are in pan-plagiarism-corpus-2011/intrinsic-detection-corpus/suspicious-document/
"""
import os
import json
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

from preprocessing import RegexTokenizer, load_nltk_stopwords
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import Ranker, BM25

try:
    from sbert_ranker import SBERTRanker, extract_sentences, hybrid_rank
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: SBERT ranker not available. Install sentence-transformers for semantic similarity.")

try:
    from jaccard_ranker import JaccardRanker
    JACCARD_AVAILABLE = True
except ImportError:
    JACCARD_AVAILABLE = False
    print("Warning: Jaccard ranker not available.")


class Logger:
    """
    Simple logger that writes to both console and file.
    """
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.log_handle = open(log_file, 'w', encoding='utf-8')
        self.log_handle.write(f"Plagiarism Detection Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write("=" * 80 + "\n\n")
        self.log_handle.flush()
    
    def log(self, message: str, flush: bool = True):
        """
        Log a message to both console and file.
        
        Args:
            message: Message to log
            flush: Whether to flush the file immediately
        """
        # Print to console
        print(message)
        # Write to file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.log_handle.write(f"[{timestamp}] {message}\n")
        if flush:
            self.log_handle.flush()
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """
        Log an error message.
        
        Args:
            message: Error message
            exception: Optional exception object
        """
        error_msg = f"ERROR: {message}"
        if exception:
            error_msg += f"\nException: {type(exception).__name__}: {str(exception)}"
        self.log(error_msg)
    
    def close(self):
        """Close the log file."""
        self.log_handle.write(f"\nLog ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.close()


def count_lines_in_file(file_path: Path) -> int:
    """
    Count the number of lines in a file efficiently.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of lines in the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def extract_document_number(doc_name: str) -> Optional[str]:
    """
    Extract document number from filename.
    For example: "source-document01007" -> "01007", "suspicious-document01007" -> "01007"
    
    Args:
        doc_name: Document filename (without extension)
        
    Returns:
        Document number string, or None if not found
    """
    # Try to extract number pattern (e.g., 01007 from source-document01007)
    match = re.search(r'(\d{5,})', doc_name)
    if match:
        return match.group(1)
    return None


def load_documents_from_text_folder(text_folder: str, logger: Logger, recursive: bool = True, max_docs: int = None, 
                                     max_lines: int = None, filter_doc_numbers: set = None) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load text documents from the text folder, optionally recursively through partX subdirectories.
    
    Args:
        text_folder: Path to the folder containing text files (or partX subdirectories)
        logger: Logger instance for output
        recursive: If True, search recursively through partX subdirectories
        max_docs: Maximum number of documents to load (None = load all)
        max_lines: Maximum number of lines per document (None = no limit)
        filter_doc_numbers: Set of document numbers to include (None = include all)
        
    Returns:
        Tuple of (doc_texts, doc_id_mapping) where:
        - doc_texts: Dictionary mapping numeric doc_id to document text
        - doc_id_mapping: Dictionary mapping numeric doc_id to original filename
    """
    doc_texts = {}
    doc_id_mapping = {}
    text_path = Path(text_folder)
    
    if not text_path.exists():
        logger.log_error(f"Text folder not found: {text_folder}")
        raise FileNotFoundError(f"Text folder not found: {text_folder}")
    
    # Get all .txt files, recursively if needed
    if recursive:
        # Search recursively through partX subdirectories
        txt_files = sorted(text_path.rglob("*.txt"))
    else:
        # Only search in the immediate directory
        txt_files = sorted(text_path.glob("*.txt"))
    
    logger.log(f"Found {len(txt_files)} text files")
    
    # Filter by document numbers if specified
    if filter_doc_numbers is not None:
        filtered_files = []
        for txt_file in txt_files:
            doc_name = txt_file.stem
            doc_number = extract_document_number(doc_name)
            if doc_number in filter_doc_numbers:
                filtered_files.append(txt_file)
        txt_files = filtered_files
        logger.log(f"Filtered to {len(txt_files)} files matching document numbers")
    
    # Use sequential IDs starting from 1 for consistency
    doc_id_counter = 1
    error_count = 0
    filtered_by_lines = 0
    
    for txt_file in txt_files:
        # Extract document ID from filename (e.g., source-document00001.txt -> source-document00001)
        doc_name = txt_file.stem
        
        # Check line count if max_lines is specified
        if max_lines is not None:
            line_count = count_lines_in_file(txt_file)
            if line_count > max_lines:
                filtered_by_lines += 1
                logger.log(f"Skipping {doc_name} ({line_count} lines > {max_lines} max)")
                continue
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:  # Only add non-empty documents
                    doc_texts[doc_id_counter] = text
                    doc_id_mapping[doc_id_counter] = doc_name
                    doc_id_counter += 1
                else:
                    logger.log(f"Warning: Skipping empty file {doc_name}")
        except Exception as e:
            error_count += 1
            logger.log_error(f"Error reading {txt_file}", e)
            continue
        
        # Limit to max_docs if specified (after filtering)
        if max_docs is not None and max_docs > 0 and len(doc_texts) >= max_docs:
            break
    
    logger.log(f"Successfully loaded {len(doc_texts)} documents")
    if filtered_by_lines > 0:
        logger.log(f"Filtered out {filtered_by_lines} documents exceeding {max_lines} lines")
    if error_count > 0:
        logger.log(f"Warning: Failed to load {error_count} files")
    return doc_texts, doc_id_mapping


def create_jsonl_from_documents(doc_texts: Dict[int, str], output_path: str, logger: Logger) -> None:
    """
    Convert documents dictionary to JSONL format required by Indexer.
    
    Args:
        doc_texts: Dictionary mapping doc_id to text
        output_path: Path to output JSONL file
        logger: Logger instance for output
    """
    logger.log(f"Creating JSONL file with {len(doc_texts)} documents...")
    doc_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, text in doc_texts.items():
            doc = {
                'docid': doc_id,
                'text': text
            }
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            doc_count += 1
            if doc_count % 50 == 0:
                logger.log(f"  Written {doc_count}/{len(doc_texts)} documents to JSONL")
    logger.log(f"JSONL file created successfully with {doc_count} documents")


def load_stopwords(stopwords_file: str = None, logger: Logger = None, use_nltk: bool = True, 
                   language: str = 'english') -> set:
    """
    Load stopwords from a text file or NLTK corpus.
    
    Args:
        stopwords_file: Path to the stopwords file (space-separated words). 
                       If None or file doesn't exist, uses NLTK stopwords.
        logger: Logger instance for output (optional)
        use_nltk: If True, use NLTK stopwords (default: True). 
                  If False and file doesn't exist, raises FileNotFoundError.
        language: Language code for NLTK stopwords (default: 'english')
        
    Returns:
        Set of stopwords as lowercase strings
        
    Raises:
        FileNotFoundError: If stopwords_file is provided but doesn't exist and use_nltk=False
        LookupError: If NLTK stopwords cannot be loaded
    """
    # Try to load from file if provided
    if stopwords_file:
        stopwords_path = Path(stopwords_file)
        
        if stopwords_path.exists():
            try:
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Split by whitespace and convert to set
                    stopwords = set(word.strip().lower() for word in content.split() if word.strip())
                    if logger:
                        logger.log(f"Loaded {len(stopwords)} stopwords from {stopwords_file}")
                    return stopwords
            except Exception as e:
                if logger:
                    logger.log_error(f"Error reading stopwords file {stopwords_file}", e)
                if not use_nltk:
                    raise
                # Fall through to NLTK if file load failed
    
    # Use NLTK stopwords (either as primary choice or fallback)
    if use_nltk:
        try:
            stopwords = load_nltk_stopwords(language=language)
            if logger:
                logger.log(f"Loaded {len(stopwords)} stopwords from NLTK ({language})")
            return stopwords
        except LookupError as e:
            if logger:
                logger.log_error(f"Failed to load NLTK stopwords: {e}")
            raise
    
    # If we get here, neither file nor NLTK worked
    if stopwords_file:
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_file} and NLTK fallback disabled")
    else:
        raise ValueError("Either stopwords_file must be provided or use_nltk must be True")


def extract_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Extract overlapping chunks of text from a document.
    This helps detect plagiarism in specific sections of papers.
    
    Args:
        text: The full text of the document
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        List of text chunks
    """
    # Split text into sentences first for better chunk boundaries
    sentences = re.split(r'[.!?]\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        sentence_word_count = len(words)
        
        if current_word_count + sentence_word_count > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + words
            current_word_count = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_word_count += sentence_word_count
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def find_best_matching_chunk(
    source_chunk: str, 
    target_text: str, 
    chunk_size: int = 200, 
    overlap: int = 50,
    use_sbert: bool = False,
    sbert_ranker: Optional[object] = None,
    use_jaccard: bool = False,
    jaccard_ranker: Optional[object] = None
) -> Tuple[int, str, float]:
    """
    Find the chunk in target_text that best matches source_chunk.
    
    Args:
        source_chunk: The source chunk text from paper A
        target_text: The full text of the target document (paper B)
        chunk_size: Size of chunks to extract
        overlap: Overlap between chunks
        use_sbert: If True, use SBERT for semantic similarity
        sbert_ranker: SBERTRanker instance (required if use_sbert=True)
        use_jaccard: If True, use Jaccard ranker (overrides default Jaccard)
        jaccard_ranker: JaccardRanker instance (optional, creates default if not provided)
        
    Returns:
        Tuple of (best_chunk_idx, best_chunk_text, similarity_score)
    """
    target_chunks = extract_chunks(target_text, chunk_size=chunk_size, overlap=overlap)
    
    # Filter out very short chunks
    valid_chunks = [(idx, chunk) for idx, chunk in enumerate(target_chunks) 
                    if len(chunk.split()) >= 20]
    
    if not valid_chunks:
        return (0, target_chunks[0] if target_chunks else "", 0.0)
    
    chunk_indices, chunks = zip(*valid_chunks)
    
    if use_sbert and sbert_ranker is not None:
        # Use SBERT for semantic similarity
        best_idx, best_text, best_score = sbert_ranker.find_best_matching_chunk_semantic(
            source_chunk,
            list(chunks),
            similarity_threshold=0.65
        )
        # Map back to original index
        actual_idx = chunk_indices[best_idx]
        return (actual_idx, best_text, best_score)
    elif use_jaccard or jaccard_ranker:
        # Use Jaccard ranker for lexical similarity
        if jaccard_ranker is None:
            jaccard_ranker = JaccardRanker() if JACCARD_AVAILABLE else None
        
        if jaccard_ranker:
            best_idx, best_text, best_score = jaccard_ranker.find_best_matching_chunk_jaccard(
                source_chunk,
                list(chunks),
                similarity_threshold=0.4
            )
            # Map back to original index
            actual_idx = chunk_indices[best_idx]
            return (actual_idx, best_text, best_score)
    
    # Fallback to simple Jaccard similarity (legacy method)
    source_words = set(word.lower() for word in source_chunk.split())
    
    best_match_idx = 0
    best_match_score = 0.0
    best_match_text = ""
    
    for idx, target_chunk in valid_chunks:
        target_words = set(word.lower() for word in target_chunk.split())
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(source_words & target_words)
        union = len(source_words | target_words)
        similarity = intersection / union if union > 0 else 0.0
        
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_idx = idx
            best_match_text = target_chunk
    
    return (best_match_idx, best_match_text, best_match_score)


def detect_plagiarism_suspicious_vs_source(
    index: BasicInvertedIndex,
    ranker: Ranker,
    source_doc_texts: Dict[int, str],
    source_doc_id_mapping: Dict[int, str],
    suspicious_doc_texts: Dict[int, str],
    suspicious_doc_id_mapping: Dict[int, str],
    logger: Logger,
    similarity_threshold: float = 0.3,
    top_k: int = 20
) -> Dict[str, List[Tuple[str, int, str, float, int, str]]]:
    """
    Detect plagiarism by querying chunks from suspicious documents against the source document index.
    Uses BM25 to find semantically similar content with shared keywords/terms.
    
    Args:
        index: The inverted index built from source documents
        ranker: The ranker to use for querying (indexed on source documents)
        source_doc_texts: Dictionary mapping source doc_id to full text
        source_doc_id_mapping: Mapping from numeric source doc_id to original filename
        suspicious_doc_texts: Dictionary mapping suspicious doc_id to full text
        suspicious_doc_id_mapping: Mapping from numeric suspicious doc_id to original filename
        logger: Logger instance for output
        similarity_threshold: Minimum BM25 score to consider a match
        top_k: Number of top BM25 results to consider per chunk
        
    Returns:
        Dictionary mapping suspicious document ID to list of tuples:
        (matched_source_doc_name, suspicious_chunk_idx, suspicious_chunk_text, bm25_score, 
         matched_source_chunk_idx, matched_source_chunk_text)
    """
    plagiarism_results = defaultdict(list)
    
    total_suspicious_docs = len(suspicious_doc_texts)
    logger.log(f"Detecting plagiarism: querying {total_suspicious_docs} suspicious documents against source index...")
    logger.log(f"Source documents in index: {len(source_doc_texts)}")
    logger.log(f"BM25 threshold: {similarity_threshold}, Top-K: {top_k}")
    
    doc_idx = 0
    for suspicious_doc_id, suspicious_text in suspicious_doc_texts.items():
        doc_idx += 1
        suspicious_doc_name = suspicious_doc_id_mapping.get(suspicious_doc_id, str(suspicious_doc_id))
        logger.log(f"\n[{doc_idx}/{total_suspicious_docs}] Processing suspicious document: {suspicious_doc_name}")
        
        try:
            # Extract chunks from this suspicious document
            chunks = extract_chunks(suspicious_text, chunk_size=200, overlap=50)
            logger.log(f"  Extracted {len(chunks)} chunks")
            
            chunks_processed = 0
            matches_found = 0
            
            # Query each chunk against the source index
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.split()) < 20:  # Skip very short chunks
                    continue
                
                chunks_processed += 1
                
                try:
                    # Query this chunk using BM25 against source documents
                    results = ranker.query(chunk)
                    
                    # Get top candidates from BM25 that exceed threshold
                    candidates = [
                        (matched_source_doc_id, bm25_score)
                        for matched_source_doc_id, bm25_score in results
                        if bm25_score >= similarity_threshold
                    ][:top_k]
                    
                    if not candidates:
                        continue
                    
                    # Add matches to results
                    for matched_source_doc_id, bm25_score in candidates:
                        matched_source_text = source_doc_texts.get(matched_source_doc_id, "")
                        if not matched_source_text:
                            continue
                        
                        # Find best matching chunk from matched source document
                        matched_chunk_idx, matched_chunk_text, _ = find_best_matching_chunk(
                            chunk, matched_source_text
                        )
                        
                        matched_source_doc_name = source_doc_id_mapping.get(matched_source_doc_id, str(matched_source_doc_id))
                        plagiarism_results[suspicious_doc_name].append((
                            matched_source_doc_name,
                            chunk_idx,  # Suspicious chunk index
                            chunk,      # Suspicious chunk text
                            bm25_score,  # BM25 similarity score
                            matched_chunk_idx,  # Matched source chunk index
                            matched_chunk_text  # Matched source chunk text
                        ))
                        matches_found += 1
                    
                except Exception as e:
                    logger.log_error(f"Error querying chunk {chunk_idx} from {suspicious_doc_name}", e)
                    continue
            
            logger.log(f"  Processed {chunks_processed} chunks, found {matches_found} potential matches")
            
        except Exception as e:
            logger.log_error(f"Error processing suspicious document {suspicious_doc_name}", e)
            continue
    
    logger.log(f"\nPlagiarism detection complete. Found matches in {len(plagiarism_results)} suspicious documents")
    return dict(plagiarism_results)


def generate_report(plagiarism_results: Dict[str, List[Tuple[str, int, str, float, int, str]]], 
                   output_file: str = "plagiarism_report.txt",
                   logger: Logger = None,
                   similarity_threshold: float = 0.3) -> None:
    """
    Generate a human-readable plagiarism detection report with yes/no answers and examples.
    
    Args:
        plagiarism_results: Results from detect_plagiarism_suspicious_vs_source
            Each match is a tuple: (matched_source_doc_name, suspicious_chunk_idx, suspicious_chunk_text, 
                                   bm25_score, matched_source_chunk_idx, matched_source_chunk_text)
        output_file: Path to output report file
        logger: Logger instance for output
        similarity_threshold: Threshold used for detection
    """
    if logger:
        logger.log(f"Generating plagiarism report: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PLAGIARISM DETECTION REPORT\n")
        f.write("PAN Plagiarism Corpus 2011 Evaluation\n")
        f.write("Using BM25 Similarity\n")
        f.write("=" * 80 + "\n\n")
        
        total_matches = sum(len(matches) for matches in plagiarism_results.values())
        f.write(f"Total suspicious documents analyzed: {len(plagiarism_results)}\n")
        f.write(f"Total potential plagiarism matches: {total_matches}\n")
        f.write(f"Threshold: {similarity_threshold} (score > {similarity_threshold} = YES, otherwise NO)\n\n")
        
        # Sort documents by number of matches (descending)
        sorted_docs = sorted(
            plagiarism_results.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for suspicious_doc_name, matches in sorted_docs:
            if not matches:
                continue
                
            f.write("-" * 80 + "\n")
            f.write(f"Suspicious Document: {suspicious_doc_name}\n")
            f.write(f"Number of potential plagiarism matches: {len(matches)}\n")
            
            # YES/NO answer based on threshold
            has_reuse = len(matches) > 0
            f.write(f"REUSE DETECTED: {'YES' if has_reuse else 'NO'}\n")
            f.write("-" * 80 + "\n\n")
            
            # Group matches by matched source document
            matches_by_source = defaultdict(list)
            for matched_source_doc, suspicious_chunk_idx, suspicious_chunk, bm25_score, matched_source_chunk_idx, matched_source_chunk in matches:
                matches_by_source[matched_source_doc].append((
                    suspicious_chunk_idx, suspicious_chunk, bm25_score, matched_source_chunk_idx, matched_source_chunk
                ))
            
            for matched_source_doc, source_matches in sorted(
                matches_by_source.items(),
                key=lambda x: max(bm25_score for _, _, bm25_score, _, _ in x[1]),
                reverse=True
            ):
                max_bm25 = max(bm25_score for _, _, bm25_score, _, _ in source_matches)
                avg_bm25 = sum(bm25_score for _, _, bm25_score, _, _ in source_matches) / len(source_matches)
                
                f.write(f"  Matches with source document: {matched_source_doc}\n")
                f.write(f"  REUSE DETECTED: YES (score: {max_bm25:.4f} > {similarity_threshold})\n")
                f.write(f"  Number of similar sections: {len(source_matches)}\n")
                f.write(f"  Highest BM25 score: {max_bm25:.4f}\n")
                f.write(f"  Average BM25 score: {avg_bm25:.4f}\n")
                f.write("\n")
                
                # Display chunk pairs as examples, sorted by BM25 score (highest first)
                sorted_chunk_matches = sorted(
                    source_matches,
                    key=lambda x: x[2],  # Sort by bm25_score
                    reverse=True
                )
                
                f.write(f"  EXAMPLES OF REUSED PASSAGES:\n")
                f.write(f"  {'=' * 76}\n\n")
                
                # Limit to top 5 examples per document pair
                for idx, (suspicious_chunk_idx, suspicious_chunk, bm25_score, matched_source_chunk_idx, matched_source_chunk) in enumerate(sorted_chunk_matches[:5]):
                    f.write(f"    Example #{idx + 1} (BM25 Score: {bm25_score:.4f})\n")
                    f.write(f"    {'-' * 76}\n")
                    f.write(f"    Suspicious Document: {suspicious_doc_name}, Chunk #{suspicious_chunk_idx}\n")
                    f.write(f"    {suspicious_chunk[:500]}{'...' if len(suspicious_chunk) > 500 else ''}\n")
                    f.write(f"\n    Source Document: {matched_source_doc}, Chunk #{matched_source_chunk_idx}\n")
                    f.write(f"    {matched_source_chunk[:500]}{'...' if len(matched_source_chunk) > 500 else ''}\n")
                    f.write("\n")
                
                if len(sorted_chunk_matches) > 5:
                    f.write(f"    ... and {len(sorted_chunk_matches) - 5} more examples (showing top 5)\n\n")
                
                f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    if logger:
        logger.log(f"Report saved to: {output_file}")
    else:
        print(f"\nReport saved to: {output_file}")


def main():
    """
    Main function to run plagiarism detection pipeline for PAN corpus.
    """
    # Configuration - using only part3 folders for faster processing
    source_folder = "pan-plagiarism-corpus-2011/external-detection-corpus/source-document/part3"
    suspicious_folder = "pan-plagiarism-corpus-2011/intrinsic-detection-corpus/suspicious-document/part3"
    stopwords_file = "stopwords.txt"
    index_cache_dir = "pan_plagiarism_index_cache_part3_filtered"
    log_file = "pan_plagiarism_detection_part3_filtered.log"
    similarity_threshold = 0.5  # BM25 threshold
    sbert_threshold = 0.65  # SBERT threshold
    jaccard_threshold = 0.4  # Jaccard threshold
    min_word_overlap_ratio = 0.3  # Minimum word overlap ratio
    top_k = 20  # Number of top BM25 matches to consider per chunk
    
    # Initialize logger
    logger = Logger(log_file)
    
    try:
        logger.log("=" * 80)
        logger.log("PAN PLAGIARISM DETECTION SYSTEM")
        logger.log("=" * 80)
        
        # Step 1: Load source documents (filter out documents with > 2500 lines)
        logger.log("\nStep 1: Loading source documents...")
        logger.log("Filtering out documents with more than 2500 lines...")
        try:
            source_doc_texts, source_doc_id_mapping = load_documents_from_text_folder(
                source_folder, logger, recursive=False, max_lines=2500
            )
        except Exception as e:
            logger.log_error("Failed to load source documents", e)
            raise
        
        if not source_doc_texts:
            logger.log("No source documents found. Exiting.")
            return
        
        # Step 2: Load suspicious documents (filter out documents with > 2500 lines)
        logger.log("\nStep 2: Loading suspicious documents...")
        logger.log("Filtering out documents with more than 2500 lines...")
        try:
            suspicious_doc_texts, suspicious_doc_id_mapping = load_documents_from_text_folder(
                suspicious_folder, logger, recursive=False, max_lines=2500
            )
        except Exception as e:
            logger.log_error("Failed to load suspicious documents", e)
            raise
        
        if not suspicious_doc_texts:
            logger.log("No suspicious documents found. Exiting.")
            return
        
        # Step 3: Create temporary JSONL file for indexing (source documents only)
        logger.log("\nStep 3: Preparing source documents for indexing...")
        temp_jsonl = None
        temp_jsonl_path = None
        try:
            temp_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
            temp_jsonl_path = temp_jsonl.name
            temp_jsonl.close()
            
            create_jsonl_from_documents(source_doc_texts, temp_jsonl_path, logger)
            logger.log(f"Created temporary JSONL file: {temp_jsonl_path}")
        except Exception as e:
            logger.log_error("Failed to create JSONL file", e)
            raise
        
        # Step 4: Initialize tokenizer and stopwords
        logger.log("\nStep 4: Initializing tokenizer and loading stopwords...")
        try:
            tokenizer = RegexTokenizer(r'\w+', lowercase=True)
            
            # Load stopwords (from file if available, otherwise from NLTK)
            stopwords = load_stopwords(stopwords_file, logger, use_nltk=True)
            logger.log(f"Initialized tokenizer with {len(stopwords)} stopwords")
        except Exception as e:
            logger.log_error("Failed to initialize tokenizer or load stopwords", e)
            raise
        
        # Step 5: Create or load index (from source documents only)
        logger.log("\nStep 5: Building index from source documents...")
        index = None
        
        try:
            # Check if index cache exists
            if os.path.exists(index_cache_dir) and os.path.exists(
                os.path.join(index_cache_dir, 'index.json')
            ):
                logger.log("Loading index from cache...")
                index = BasicInvertedIndex()
                index.load(index_cache_dir)
                stats = index.get_statistics()
                logger.log(f"Index loaded successfully")
                logger.log(f"  - Documents indexed: {stats.get('number_of_documents', 0)}")
                logger.log(f"  - Vocabulary size: {stats.get('unique_token_count', 0)}")
            else:
                logger.log("Creating new index (this may take a while)...")
                logger.log(f"  - Dataset: {temp_jsonl_path}")
                logger.log(f"  - Min word frequency: 2")
                logger.log(f"  - Stopwords: {len(stopwords)}")
                
                index = Indexer.create_index(
                    index_type=IndexType.BasicInvertedIndex,
                    dataset_path=temp_jsonl_path,
                    document_preprocessor=tokenizer,
                    stopwords=stopwords,
                    minimum_word_frequency=2,  # Filter out very rare words
                    text_key='text',
                    max_docs=-1,  # Process all documents
                    id_key='docid'
                )
                
                stats = index.get_statistics()
                logger.log(f"Index created successfully")
                logger.log(f"  - Documents indexed: {stats.get('number_of_documents', 0)}")
                logger.log(f"  - Vocabulary size: {stats.get('unique_token_count', 0)}")
                
                logger.log("Saving index to cache...")
                os.makedirs(index_cache_dir, exist_ok=True)
                index.save(index_cache_dir)
                logger.log(f"Index saved to: {index_cache_dir}")
        except Exception as e:
            logger.log_error("Failed to build/load index", e)
            raise
        
        # Step 6: Initialize ranker (indexed on source documents)
        logger.log("\nStep 6: Initializing ranker...")
        try:
            scorer = BM25(index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8})
            ranker = Ranker(
                index=index,
                document_preprocessor=tokenizer,
                stopwords=stopwords,
                scorer=scorer,
                raw_text_dict=source_doc_texts  # Use source documents for raw text lookup
            )
            logger.log("Ranker initialized successfully (BM25)")
        except Exception as e:
            logger.log_error("Failed to initialize ranker", e)
            raise
        
        # Step 7: Detect plagiarism (query suspicious documents against source index)
        logger.log("\nStep 7: Running plagiarism detection...")
        try:
            plagiarism_results = detect_plagiarism_suspicious_vs_source(
                index=index,
                ranker=ranker,
                source_doc_texts=source_doc_texts,
                source_doc_id_mapping=source_doc_id_mapping,
                suspicious_doc_texts=suspicious_doc_texts,
                suspicious_doc_id_mapping=suspicious_doc_id_mapping,
                logger=logger,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
        except Exception as e:
            logger.log_error("Failed during plagiarism detection", e)
            raise
        
        # Step 8: Save full results to JSON (for detailed analysis)
        logger.log("\nStep 8: Saving full plagiarism results to JSON...")
        try:
            # Convert results to JSON-serializable format
            json_results = {}
            for suspicious_doc, matches in plagiarism_results.items():
                json_results[suspicious_doc] = [
                    {
                        'source_doc': source_doc,
                        'suspicious_chunk_idx': suspicious_chunk_idx,
                        'source_chunk_idx': source_chunk_idx,
                        'bm25_score': float(bm25_score)
                    }
                    for source_doc, suspicious_chunk_idx, _, bm25_score, source_chunk_idx, _ in matches
                ]
            
            json_file = "pan_plagiarism_results_part3_filtered.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.log(f"Full results saved to {json_file}")
        except Exception as e:
            logger.log_error("Failed to save JSON results", e)
            # Don't raise - continue to report generation
        
        # Step 9: Generate report
        logger.log("\nStep 9: Generating report...")
        try:
            generate_report(plagiarism_results, "pan_plagiarism_report_part3_filtered.txt", logger, similarity_threshold)
        except Exception as e:
            logger.log_error("Failed to generate report", e)
            raise
        
        # Cleanup temporary file
        if temp_jsonl_path:
            try:
                os.unlink(temp_jsonl_path)
                logger.log(f"Cleaned up temporary file: {temp_jsonl_path}")
            except Exception as e:
                logger.log_error(f"Failed to cleanup temp file {temp_jsonl_path}", e)
        
        logger.log("\n" + "=" * 80)
        logger.log("PLAGIARISM DETECTION COMPLETE")
        logger.log("=" * 80)
        
        # Print summary statistics with YES/NO answers
        total_suspicious_docs = len(suspicious_doc_texts)
        total_docs_with_matches = len([d for d, matches in plagiarism_results.items() if matches])
        total_matches = sum(len(matches) for matches in plagiarism_results.values())
        
        logger.log(f"\nSummary:")
        logger.log(f"  Source documents indexed: {len(source_doc_texts)}")
        logger.log(f"  Suspicious documents analyzed: {total_suspicious_docs}")
        logger.log(f"  Suspicious documents with reuse detected (YES): {total_docs_with_matches}")
        logger.log(f"  Suspicious documents with no reuse detected (NO): {total_suspicious_docs - total_docs_with_matches}")
        logger.log(f"  Total potential matches found: {total_matches}")
        logger.log(f"  Threshold: {similarity_threshold} (score > {similarity_threshold} = YES, otherwise NO)")
        
        # Print YES/NO summary for each suspicious document
        logger.log(f"\nYES/NO Detection Summary:")
        logger.log(f"  {'Suspicious Document':<50} {'Has Reuse':<15}")
        logger.log(f"  {'-' * 50} {'-' * 15}")
        for suspicious_doc_name, matches in sorted(plagiarism_results.items()):
            has_reuse = "YES" if matches else "NO"
            logger.log(f"  {suspicious_doc_name:<50} {has_reuse:<15}")
        
        # Also include suspicious documents with no matches
        suspicious_docs_with_matches = set(plagiarism_results.keys())
        for suspicious_doc_id, suspicious_doc_name in suspicious_doc_id_mapping.items():
            if suspicious_doc_name not in suspicious_docs_with_matches:
                logger.log(f"  {suspicious_doc_name:<50} {'NO':<15}")
        
        logger.log(f"\nDetailed report with examples saved to: pan_plagiarism_report_part3_filtered.txt")
        logger.log(f"Log file saved to: {log_file}")
        
    except Exception as e:
        logger.log_error("Fatal error in main execution", e)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
