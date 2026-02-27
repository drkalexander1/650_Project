"""
Plagiarism Detection System for Academic Papers
This script uses information retrieval techniques to detect copy-pasted/rephrased
sections between papers that are not cited.
"""
import os
import json
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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


def load_documents_from_metadata(corpus_folder: str, logger: Logger) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load all text documents from the corpus folder based on metadata files.
    This function reads metadata JSON files to find which text files exist and loads them.
    
    Args:
        corpus_folder: Path to the corpus folder (e.g., "corpus_arxiv_Edward_Witten")
        logger: Logger instance for output
        
    Returns:
        Tuple of (doc_texts, doc_id_mapping) where:
        - doc_texts: Dictionary mapping numeric doc_id to document text
        - doc_id_mapping: Dictionary mapping numeric doc_id to original filename
    """
    doc_texts = {}
    doc_id_mapping = {}
    corpus_path = Path(corpus_folder)
    metadata_dir = corpus_path / "metadata_arxiv"
    
    if not corpus_path.exists():
        logger.log_error(f"Corpus folder not found: {corpus_folder}")
        raise FileNotFoundError(f"Corpus folder not found: {corpus_folder}")
    
    if not metadata_dir.exists():
        logger.log_error(f"Metadata directory not found: {metadata_dir}")
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    
    # Get all metadata JSON files
    metadata_files = sorted(metadata_dir.glob("*.json"))
    logger.log(f"Found {len(metadata_files)} metadata files")
    
    # Use sequential IDs starting from 1 for consistency
    doc_id_counter = 1
    error_count = 0
    skipped_count = 0
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Check if there are downloaded versions
            if 'versions_downloaded' not in metadata or not metadata['versions_downloaded']:
                skipped_count += 1
                continue
            
            # Load text from each version (typically we want the latest, but load all available)
            for version_info in metadata['versions_downloaded']:
                text_file_path = version_info.get('text_file', '')
                if not text_file_path:
                    continue
                
                # Handle path resolution - metadata paths are relative to project root
                text_file = None
                if Path(text_file_path).is_absolute():
                    text_file = Path(text_file_path)
                else:
                    # Try as-is (relative to current working directory)
                    text_file = Path(text_file_path)
                    if not text_file.exists():
                        # Try relative to corpus folder
                        text_file = corpus_path / Path(text_file_path).name
                    if not text_file.exists() and 'text_arxiv' in text_file_path:
                        # Try constructing path from corpus folder
                        text_file = corpus_path / "text_arxiv" / Path(text_file_path).name
                
                if not text_file or not text_file.exists():
                    logger.log(f"Warning: Text file not found: {text_file_path}")
                    continue
                
                # Extract document name from filename (e.g., 0706.3359_v1.txt -> 0706.3359_v1)
                doc_name = text_file.stem
                
                try:
                    with open(text_file, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
                        if text:  # Only add non-empty documents
                            doc_texts[doc_id_counter] = text
                            doc_id_mapping[doc_id_counter] = doc_name
                            doc_id_counter += 1
                        else:
                            logger.log(f"Warning: Skipping empty file {doc_name}")
                except Exception as e:
                    error_count += 1
                    logger.log_error(f"Error reading {text_file}", e)
                    continue
                    
        except Exception as e:
            error_count += 1
            logger.log_error(f"Error reading metadata file {metadata_file}", e)
            continue
    
    logger.log(f"Successfully loaded {len(doc_texts)} documents from metadata")
    if skipped_count > 0:
        logger.log(f"Skipped {skipped_count} metadata files with no downloaded versions")
    if error_count > 0:
        logger.log(f"Warning: Failed to load {error_count} files")
    return doc_texts, doc_id_mapping


def load_documents_from_text_folder(text_folder: str, logger: Logger) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load all text documents from the text folder.
    
    Args:
        text_folder: Path to the folder containing text files
        logger: Logger instance for output
        
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
    
    # Get all .txt files and sort for consistent ordering
    txt_files = sorted(text_path.glob("*.txt"))
    logger.log(f"Found {len(txt_files)} text files")
    
    # Use sequential IDs starting from 1 for consistency
    doc_id_counter = 1
    error_count = 0
    
    for txt_file in txt_files:
        # Extract document ID from filename (e.g., PMC10000860.txt -> PMC10000860)
        doc_name = txt_file.stem
        
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
    
    logger.log(f"Successfully loaded {len(doc_texts)} documents")
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


def has_sufficient_overlap(chunk1: str, chunk2: str, min_overlap_ratio: float = 0.3) -> bool:
    """
    Check if chunks have sufficient word overlap to be considered similar.
    
    Args:
        chunk1: First text chunk
        chunk2: Second text chunk
        min_overlap_ratio: Minimum ratio of overlapping words to smallest chunk size
        
    Returns:
        True if chunks have sufficient overlap, False otherwise
    """
    words1 = set(word.lower() for word in chunk1.split())
    words2 = set(word.lower() for word in chunk2.split())
    
    if len(words1) == 0 or len(words2) == 0:
        return False
    
    overlap = len(words1 & words2)
    min_words = min(len(words1), len(words2))
    
    return (overlap / min_words) >= min_overlap_ratio


def find_best_matching_chunk(
    source_chunk: str, 
    target_text: str, 
    chunk_size: int = 200, 
    overlap: int = 50,
    use_sbert: bool = False,
    sbert_ranker: Optional[object] = None,
    use_jaccard: bool = False,
    jaccard_ranker: Optional[object] = None,
    sbert_threshold: float = 0.65,
    jaccard_threshold: float = 0.4
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
        sbert_threshold: Minimum SBERT similarity threshold (default: 0.65)
        jaccard_threshold: Minimum Jaccard similarity threshold (default: 0.4)
        
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
        # Use SBERT for semantic similarity with stricter threshold
        best_idx, best_text, best_score = sbert_ranker.find_best_matching_chunk_semantic(
            source_chunk,
            list(chunks),
            similarity_threshold=sbert_threshold
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
                similarity_threshold=jaccard_threshold
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


def detect_plagiarism(
    index: BasicInvertedIndex,
    ranker: Ranker,
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    similarity_threshold: float = 0.5,
    top_k: int = 10,
    use_sbert: bool = False,
    sbert_ranker: Optional[object] = None,
    use_jaccard: bool = False,
    jaccard_ranker: Optional[object] = None,
    hybrid_mode: bool = False,
    bm25_weight: float = 0.5,
    sbert_weight: float = 0.5,
    jaccard_weight: float = 0.0,
    sbert_threshold: float = 0.65,
    jaccard_threshold: float = 0.4,
    min_word_overlap_ratio: float = 0.3
) -> Dict[str, List[Tuple[str, int, str, float, int, str, Tuple[float, float]]]]:
    """
    Detect plagiarism by querying chunks from each document against the index.
    
    Args:
        index: The inverted index
        ranker: The ranker to use for querying
        doc_texts: Dictionary mapping doc_id to full text
        doc_id_mapping: Mapping from numeric doc_id to original filename
        logger: Logger instance for output
        similarity_threshold: Minimum BM25 similarity score to consider as plagiarism (default: 0.5)
        top_k: Number of top results to consider per chunk
        sbert_threshold: Minimum SBERT similarity threshold (default: 0.65)
        jaccard_threshold: Minimum Jaccard similarity threshold (default: 0.4)
        min_word_overlap_ratio: Minimum word overlap ratio between chunks (default: 0.3)
        
    Returns:
        Dictionary mapping document ID to list of tuples:
        (matched_doc_name, source_chunk_idx, source_chunk_text, score, matched_chunk_idx, matched_chunk_text, (sbert_score, jaccard_score))
    """
    plagiarism_results = defaultdict(list)
    
    total_docs = len(doc_texts)
    logger.log(f"Detecting plagiarism across {total_docs} documents...")
    logger.log(f"BM25 threshold: {similarity_threshold}, SBERT threshold: {sbert_threshold}, Jaccard threshold: {jaccard_threshold}, Top-K: {top_k}")
    logger.log(f"Minimum word overlap ratio: {min_word_overlap_ratio}")
    
    # Configure ranking methods
    ranking_methods = []
    if use_sbert:
        if sbert_ranker is None:
            logger.log("Warning: use_sbert=True but no sbert_ranker provided. Falling back to lexical matching.")
            use_sbert = False
        else:
            logger.log(f"Using SBERT semantic similarity ranking")
            ranking_methods.append("SBERT")
    
    if use_jaccard:
        if jaccard_ranker is None:
            if JACCARD_AVAILABLE:
                jaccard_ranker = JaccardRanker()
                logger.log(f"Initialized default Jaccard ranker")
            else:
                logger.log("Warning: use_jaccard=True but Jaccard ranker not available.")
                use_jaccard = False
        if use_jaccard:
            logger.log(f"Using Jaccard lexical similarity ranking")
            ranking_methods.append("Jaccard")
    
    if hybrid_mode and len(ranking_methods) > 0:
        weights_desc = []
        if bm25_weight > 0:
            weights_desc.append(f"BM25={bm25_weight}")
        if sbert_weight > 0:
            weights_desc.append(f"SBERT={sbert_weight}")
        if jaccard_weight > 0:
            weights_desc.append(f"Jaccard={jaccard_weight}")
        logger.log(f"Hybrid mode: Combining {' + '.join(weights_desc)}")
    
    doc_idx = 0
    for doc_id, text in doc_texts.items():
        doc_idx += 1
        doc_name = doc_id_mapping.get(doc_id, str(doc_id))
        logger.log(f"\n[{doc_idx}/{total_docs}] Processing document: {doc_name}")
        
        try:
            # Extract chunks from this document
            chunks = extract_chunks(text, chunk_size=200, overlap=50)
            logger.log(f"  Extracted {len(chunks)} chunks")
            
            chunks_processed = 0
            matches_found = 0
            
            # Query each chunk against the index
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.split()) < 20:  # Skip very short chunks
                    continue
                
                chunks_processed += 1
                
                try:
                    # Query this chunk using BM25
                    bm25_results = ranker.query(chunk)
                    
                    # Filter results: exclude the document itself and low-scoring matches
                    bm25_filtered = [
                        (matched_doc_id, score)
                        for matched_doc_id, score in bm25_results
                        if matched_doc_id != doc_id and score >= similarity_threshold
                    ][:top_k]
                    
                    # If using SBERT or Jaccard, also compute additional similarity scores
                    sbert_scores = []
                    jaccard_scores = []
                    
                    if use_sbert and sbert_ranker:
                        # Get candidate documents from BM25 results
                        candidate_doc_ids = [doc_id for doc_id, _ in bm25_filtered[:top_k * 2]]  # Get more candidates
                        
                        # Compute SBERT scores for candidate chunks
                        for matched_doc_id in candidate_doc_ids:
                            matched_text = doc_texts.get(matched_doc_id, "")
                            if matched_text:
                                # Find best matching chunk using SBERT with stricter threshold
                                _, _, sbert_score = find_best_matching_chunk(
                                    chunk, matched_text, use_sbert=True, sbert_ranker=sbert_ranker,
                                    sbert_threshold=sbert_threshold
                                )
                                # Use stricter SBERT threshold
                                if sbert_score >= sbert_threshold:
                                    sbert_scores.append((matched_doc_id, sbert_score))
                    
                    if use_jaccard and jaccard_ranker:
                        # Get candidate documents from BM25 results
                        candidate_doc_ids = [doc_id for doc_id, _ in bm25_filtered[:top_k * 2]]  # Get more candidates
                        
                        # Compute Jaccard scores for candidate chunks
                        for matched_doc_id in candidate_doc_ids:
                            matched_text = doc_texts.get(matched_doc_id, "")
                            if matched_text:
                                # Find best matching chunk using Jaccard with stricter threshold
                                _, _, jaccard_score = find_best_matching_chunk(
                                    chunk, matched_text, use_jaccard=True, jaccard_ranker=jaccard_ranker,
                                    jaccard_threshold=jaccard_threshold
                                )
                                # Use stricter Jaccard threshold
                                if jaccard_score >= jaccard_threshold:
                                    jaccard_scores.append((matched_doc_id, jaccard_score))
                    
                    # Determine final results based on mode
                    if hybrid_mode and (use_sbert or use_jaccard):
                        # Hybrid ranking: combine multiple methods
                        from sbert_ranker import hybrid_rank
                        
                        # Start with BM25 scores
                        combined_scores = bm25_filtered
                        
                        # Add SBERT scores if available
                        if sbert_scores:
                            combined_scores = hybrid_rank(
                                combined_scores,
                                sbert_scores,
                                bm25_weight=bm25_weight,
                                sbert_weight=sbert_weight
                            )
                        
                        # Add Jaccard scores if available (normalize weights)
                        if jaccard_scores and jaccard_weight > 0:
                            # Normalize weights: adjust BM25 and SBERT weights, add Jaccard
                            total_weight = bm25_weight + sbert_weight + jaccard_weight
                            if total_weight > 0:
                                # Combine with normalized weights
                                jaccard_norm = jaccard_weight / total_weight
                                
                                # Convert to dicts for easier combination
                                combined_dict = {doc_id: score for doc_id, score in combined_scores}
                                jaccard_dict = {doc_id: score for doc_id, score in jaccard_scores}
                                
                                # Find max scores for normalization
                                combined_max = max(combined_dict.values()) if combined_dict else 1.0
                                jaccard_max = max(jaccard_dict.values()) if jaccard_dict else 1.0
                                
                                # Get all doc IDs
                                all_doc_ids = set(combined_dict.keys()) | set(jaccard_dict.keys())
                                
                                # Combine scores
                                final_scores = []
                                for doc_id in all_doc_ids:
                                    combined_score = (combined_dict.get(doc_id, 0.0) / combined_max) if combined_max > 0 else 0.0
                                    jaccard_score = (jaccard_dict.get(doc_id, 0.0) / jaccard_max) if jaccard_max > 0 else 0.0
                                    
                                    # Weighted combination
                                    final_score = ((1 - jaccard_norm) * combined_score) + (jaccard_norm * jaccard_score)
                                    final_scores.append((doc_id, final_score))
                                
                                combined_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
                        
                        filtered_results = combined_scores[:top_k]
                    elif use_sbert and sbert_scores:
                        # Use SBERT scores only
                        filtered_results = sorted(sbert_scores, key=lambda x: x[1], reverse=True)[:top_k]
                    elif use_jaccard and jaccard_scores:
                        # Use Jaccard scores only
                        filtered_results = sorted(jaccard_scores, key=lambda x: x[1], reverse=True)[:top_k]
                    else:
                        # Use BM25 only
                        filtered_results = bm25_filtered
                    
                    if filtered_results:
                        matches_found += len(filtered_results)
                        for matched_doc_id, score in filtered_results:
                            matched_doc_name = doc_id_mapping.get(matched_doc_id, str(matched_doc_id))
                            
                            # Find the best matching chunk in the matched document
                            matched_text = doc_texts.get(matched_doc_id, "")
                            if matched_text:
                                matched_chunk_idx, matched_chunk_text, similarity_score = find_best_matching_chunk(
                                    chunk, matched_text, 
                                    use_sbert=use_sbert, sbert_ranker=sbert_ranker,
                                    use_jaccard=use_jaccard, jaccard_ranker=jaccard_ranker,
                                    sbert_threshold=sbert_threshold,
                                    jaccard_threshold=jaccard_threshold
                                )
                                
                                # Also compute individual scores for reporting
                                sbert_score = 0.0
                                jaccard_score = 0.0
                                if use_sbert and sbert_ranker:
                                    _, _, sbert_score = find_best_matching_chunk(
                                        chunk, matched_text, use_sbert=True, sbert_ranker=sbert_ranker,
                                        sbert_threshold=sbert_threshold
                                    )
                                if use_jaccard and jaccard_ranker:
                                    _, _, jaccard_score = find_best_matching_chunk(
                                        chunk, matched_text, use_jaccard=True, jaccard_ranker=jaccard_ranker,
                                        jaccard_threshold=jaccard_threshold
                                    )
                                
                                # Require multiple signals for plagiarism detection
                                # If using SBERT, require both BM25 and SBERT to exceed thresholds
                                is_valid_match = True
                                
                                if use_sbert and sbert_ranker:
                                    # Require SBERT score to also exceed threshold
                                    if sbert_score < sbert_threshold:
                                        is_valid_match = False
                                
                                # Check minimum word overlap
                                if is_valid_match and not has_sufficient_overlap(chunk, matched_chunk_text, min_word_overlap_ratio):
                                    is_valid_match = False
                                
                                # Only add if all checks pass
                                if is_valid_match:
                                    plagiarism_results[doc_name].append((
                                        matched_doc_name,
                                        chunk_idx,  # Source chunk index from paper A
                                        chunk,      # Source chunk text from paper A
                                        score,      # BM25 or combined similarity score
                                        matched_chunk_idx,  # Matched chunk index from paper B
                                        matched_chunk_text,  # Matched chunk text from paper B
                                        (sbert_score, jaccard_score)  # Tuple of (sbert_score, jaccard_score)
                                    ))
                                else:
                                    matches_found -= 1  # Adjust count since we filtered this out
                except Exception as e:
                    logger.log_error(f"Error querying chunk {chunk_idx} from {doc_name}", e)
                    continue
            
            logger.log(f"  Processed {chunks_processed} chunks, found {matches_found} potential matches")
            
        except Exception as e:
            logger.log_error(f"Error processing document {doc_name}", e)
            continue
    
    logger.log(f"\nPlagiarism detection complete. Found matches in {len(plagiarism_results)} documents")
    return dict(plagiarism_results)


def generate_report(plagiarism_results: Dict[str, List[Tuple[str, int, str, float, int, str, Tuple[float, float]]]], 
                   output_file: str = "plagiarism_report.txt",
                   logger: Logger = None,
                   similarity_threshold: float = 0.3,
                   all_doc_names: List[str] = None) -> None:
    """
    Generate a human-readable plagiarism detection report with yes/no answers and examples.
    
    Args:
        plagiarism_results: Results from detect_plagiarism
            Each match is a tuple: (matched_doc_name, source_chunk_idx, source_chunk_text, 
                                   score, matched_chunk_idx, matched_chunk_text, (sbert_score, jaccard_score))
        output_file: Path to output report file
        logger: Logger instance for output
        similarity_threshold: Threshold used for detection (default: 0.3)
        all_doc_names: Optional list of all document names to include documents with no matches
    """
    if logger:
        logger.log(f"Generating plagiarism report: {output_file}")
    
    # Get all document names (from results or provided list)
    if all_doc_names is None:
        all_doc_names = list(plagiarism_results.keys())
    else:
        # Include documents from results that might not be in all_doc_names
        all_doc_names = list(set(all_doc_names) | set(plagiarism_results.keys()))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PLAGIARISM DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        total_matches = sum(len(matches) for matches in plagiarism_results.values())
        total_docs_with_plagiarism = len([d for d, matches in plagiarism_results.items() if matches])
        f.write(f"Total documents analyzed: {len(all_doc_names)}\n")
        f.write(f"Documents with plagiarism detected (YES): {total_docs_with_plagiarism}\n")
        f.write(f"Documents with no plagiarism detected (NO): {len(all_doc_names) - total_docs_with_plagiarism}\n")
        f.write(f"Total potential plagiarism matches: {total_matches}\n")
        f.write(f"Threshold: {similarity_threshold} (score > {similarity_threshold} = YES, otherwise NO)\n\n")
        
        # Sort documents: first those with matches (by number of matches), then those without
        docs_with_matches = [(name, matches) for name, matches in plagiarism_results.items() if matches]
        docs_without_matches = [name for name in all_doc_names if name not in plagiarism_results or not plagiarism_results[name]]
        
        sorted_docs_with_matches = sorted(
            docs_with_matches,
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Process documents with matches
        for doc_name, matches in sorted_docs_with_matches:
            f.write("-" * 80 + "\n")
            f.write(f"Document: {doc_name}\n")
            f.write(f"Number of potential plagiarism matches: {len(matches)}\n")
            
            # YES/NO answer based on threshold
            has_plagiarism = len(matches) > 0
            f.write(f"PLAGIARISM DETECTED: {'YES' if has_plagiarism else 'NO'}\n")
            f.write("-" * 80 + "\n\n")
            
            # Group matches by matched document
            matches_by_doc = defaultdict(list)
            for matched_doc, source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk, similarity_scores in matches:
                matches_by_doc[matched_doc].append((
                    source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk, similarity_scores
                ))
            
            for matched_doc, doc_matches in sorted(
                matches_by_doc.items(),
                key=lambda x: max(score for _, _, score, _, _, _ in x[1]),
                reverse=True
            ):
                max_score = max(score for _, _, score, _, _, _ in doc_matches)
                avg_score = sum(score for _, _, score, _, _, _ in doc_matches) / len(doc_matches)
                
                f.write(f"  Matches with: {matched_doc}\n")
                f.write(f"  PLAGIARISM DETECTED: YES (score: {max_score:.4f} > {similarity_threshold})\n")
                f.write(f"  Number of similar sections: {len(doc_matches)}\n")
                f.write(f"  Highest similarity score: {max_score:.4f}\n")
                f.write(f"  Average similarity score: {avg_score:.4f}\n")
                f.write("\n")
                
                # Display chunk pairs as examples, sorted by similarity score (highest first)
                sorted_chunk_matches = sorted(
                    doc_matches,
                    key=lambda x: x[2],  # Sort by score
                    reverse=True
                )
                
                f.write(f"  EXAMPLES OF PLAGIARIZED PASSAGES:\n")
                f.write(f"  {'=' * 76}\n\n")
                
                # Limit to top 5 examples per document pair
                for idx, (source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk, similarity_scores) in enumerate(sorted_chunk_matches[:5]):
                    score_label = "Combined" if len(doc_matches) > 0 and len(doc_matches[0]) > 5 else "BM25"
                    f.write(f"    Example #{idx + 1} ({score_label} Score: {score:.4f}")
                    sbert_score, jaccard_score = similarity_scores
                    if sbert_score > 0:
                        f.write(f", SBERT Score: {sbert_score:.4f}")
                    if jaccard_score > 0:
                        f.write(f", Jaccard Score: {jaccard_score:.4f}")
                    f.write(")\n")
                    f.write(f"    {'-' * 76}\n")
                    f.write(f"    Source Document: {doc_name}, Chunk #{source_chunk_idx}\n")
                    f.write(f"    {source_chunk[:500]}{'...' if len(source_chunk) > 500 else ''}\n")
                    f.write(f"\n    Matched Document: {matched_doc}, Chunk #{matched_chunk_idx}\n")
                    f.write(f"    {matched_chunk[:500]}{'...' if len(matched_chunk) > 500 else ''}\n")
                    f.write("\n")
                
                if len(sorted_chunk_matches) > 5:
                    f.write(f"    ... and {len(sorted_chunk_matches) - 5} more examples (showing top 5)\n\n")
                
                f.write("\n")
        
        # Process documents without matches
        if docs_without_matches:
            f.write("-" * 80 + "\n")
            f.write("DOCUMENTS WITH NO PLAGIARISM DETECTED\n")
            f.write("-" * 80 + "\n\n")
            for doc_name in sorted(docs_without_matches):
                f.write(f"Document: {doc_name}\n")
                f.write(f"PLAGIARISM DETECTED: NO\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    if logger:
        logger.log(f"Report saved to: {output_file}")
    else:
        print(f"\nReport saved to: {output_file}")


def save_plagiarism_chunks_dict(
    plagiarism_results: Dict[str, List[Tuple[str, int, str, float, int, str, Tuple[float, float]]]],
    output_file: str,
    logger: Logger = None
) -> Dict[str, List[Dict]]:
    """
    Save all plagiarism chunks to a dictionary for statistical analysis.
    
    Args:
        plagiarism_results: Results from detect_plagiarism
        output_file: Path to output JSON file
        logger: Logger instance for output
        
    Returns:
        Dictionary mapping document names to list of plagiarism chunk dictionaries
    """
    if logger:
        logger.log(f"Saving plagiarism chunks dictionary to {output_file}...")
    
    chunks_dict = {}
    
    for doc_name, matches in plagiarism_results.items():
        chunks_dict[doc_name] = []
        for matched_doc, source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk, similarity_scores in matches:
            sbert_score, jaccard_score = similarity_scores
            chunks_dict[doc_name].append({
                'matched_doc': matched_doc,
                'source_chunk_idx': source_chunk_idx,
                'source_chunk_text': source_chunk,
                'score': float(score),
                'matched_chunk_idx': matched_chunk_idx,
                'matched_chunk_text': matched_chunk,
                'sbert_score': float(sbert_score),
                'jaccard_score': float(jaccard_score)
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
    
    if logger:
        logger.log(f"Saved {sum(len(chunks) for chunks in chunks_dict.values())} plagiarism chunks to {output_file}")
    
    return chunks_dict


def calculate_statistics(
    plagiarism_results: Dict[str, List[Tuple[str, int, str, float, int, str, Tuple[float, float]]]],
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger
) -> Dict[str, Dict]:
    """
    Calculate statistics about plagiarism detection results.
    
    Args:
        plagiarism_results: Results from detect_plagiarism
        doc_texts: Dictionary mapping doc_id to text
        doc_id_mapping: Mapping from doc_id to doc_name
        logger: Logger instance
        
    Returns:
        Dictionary mapping doc_name to statistics dict
    """
    logger.log("Calculating plagiarism statistics...")
    
    stats_dict = {}
    
    # Create reverse mapping: doc_name -> doc_id
    name_to_id = {name: doc_id for doc_id, name in doc_id_mapping.items()}
    
    for doc_name, matches in plagiarism_results.items():
        # Get document text
        doc_id = name_to_id.get(doc_name)
        if doc_id is None:
            continue
        
        doc_text = doc_texts.get(doc_id, "")
        if not doc_text:
            continue
        
        # Extract chunks
        chunks = extract_chunks(doc_text, chunk_size=200, overlap=50)
        total_chunks = len([c for c in chunks if len(c.split()) >= 20])
        
        # Count unique plagiarized chunks
        plagiarized_chunk_indices = set()
        scores = []
        sbert_scores = []
        jaccard_scores = []
        
        for matched_doc, source_chunk_idx, _, score, _, _, similarity_scores in matches:
            if source_chunk_idx < total_chunks:
                plagiarized_chunk_indices.add(source_chunk_idx)
            scores.append(score)
            sbert_score, jaccard_score = similarity_scores
            if sbert_score > 0:
                sbert_scores.append(sbert_score)
            if jaccard_score > 0:
                jaccard_scores.append(jaccard_score)
        
        plagiarized_chunks = len(plagiarized_chunk_indices)
        percentage = (plagiarized_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
        
        stats_dict[doc_name] = {
            'total_chunks': total_chunks,
            'plagiarized_chunks': plagiarized_chunks,
            'percentage': percentage,
            'num_matches': len(matches),
            'avg_score': float(np.mean(scores)) if scores else 0.0,
            'max_score': float(max(scores)) if scores else 0.0,
            'avg_sbert_score': float(np.mean(sbert_scores)) if sbert_scores else 0.0,
            'avg_jaccard_score': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
        }
    
    # Also process documents with no matches
    for doc_id, doc_name in doc_id_mapping.items():
        if doc_name not in stats_dict:
            doc_text = doc_texts.get(doc_id, "")
            if doc_text:
                chunks = extract_chunks(doc_text, chunk_size=200, overlap=50)
                total_chunks = len([c for c in chunks if len(c.split()) >= 20])
                stats_dict[doc_name] = {
                    'total_chunks': total_chunks,
                    'plagiarized_chunks': 0,
                    'percentage': 0.0,
                    'num_matches': 0,
                    'avg_score': 0.0,
                    'max_score': 0.0,
                    'avg_sbert_score': 0.0,
                    'avg_jaccard_score': 0.0
                }
    
    logger.log(f"Calculated statistics for {len(stats_dict)} documents")
    return stats_dict


def create_visualizations(
    stats_dict: Dict[str, Dict],
    output_dir: Path,
    logger: Logger
):
    """
    Create visualizations of plagiarism statistics.
    
    Args:
        stats_dict: Dictionary mapping doc_name to statistics
        output_dir: Directory to save visualizations
        logger: Logger instance
    """
    logger.log("Creating visualizations...")
    
    percentages = [stats['percentage'] for stats in stats_dict.values()]
    scores = [stats['avg_score'] for stats in stats_dict.values() if stats['avg_score'] > 0]
    
    # 1. Histogram of percentage distribution
    if percentages:
        plt.figure(figsize=(12, 6))
        plt.hist(percentages, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Percentage of Plagiarized Chunks (%)', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title('Distribution of Plagiarized Chunk Percentages', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'plagiarism_percentage_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  Saved: plagiarism_percentage_histogram.png")
    
    # 2. Histogram of similarity scores
    if scores:
        plt.figure(figsize=(12, 6))
        plt.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Average Similarity Score', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.title('Distribution of Average Similarity Scores', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_score_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  Saved: similarity_score_histogram.png")
    
    # 3. Box plot of percentages
    if percentages:
        plt.figure(figsize=(8, 6))
        plt.boxplot(percentages, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
        plt.ylabel('Percentage of Plagiarized Chunks (%)', fontsize=12)
        plt.title('Box Plot of Plagiarized Chunk Percentages', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'plagiarism_percentage_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  Saved: plagiarism_percentage_boxplot.png")
    
    # 4. Scatter plot: percentage vs number of matches
    percentages_list = [stats['percentage'] for stats in stats_dict.values()]
    num_matches_list = [stats['num_matches'] for stats in stats_dict.values()]
    
    if percentages_list and num_matches_list:
        plt.figure(figsize=(10, 6))
        plt.scatter(num_matches_list, percentages_list, alpha=0.6)
        plt.xlabel('Number of Matches', fontsize=12)
        plt.ylabel('Percentage of Plagiarized Chunks (%)', fontsize=12)
        plt.title('Plagiarized Chunk Percentage vs Number of Matches', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'percentage_vs_matches_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.log("  Saved: percentage_vs_matches_scatter.png")
    
    logger.log("All visualizations created successfully")


def save_statistics_report(
    stats_dict: Dict[str, Dict],
    output_file: str,
    logger: Logger
):
    """
    Save statistics report to a text file.
    
    Args:
        stats_dict: Dictionary mapping doc_name to statistics
        output_file: Path to output file
        logger: Logger instance
    """
    logger.log(f"Saving statistics report to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PLAGIARISM STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total documents analyzed: {len(stats_dict)}\n")
        
        # Summary statistics
        percentages = [stats['percentage'] for stats in stats_dict.values()]
        scores = [stats['avg_score'] for stats in stats_dict.values() if stats['avg_score'] > 0]
        
        if percentages:
            f.write(f"\nSummary Statistics:\n")
            f.write(f"  Mean percentage of plagiarized chunks: {np.mean(percentages):.2f}%\n")
            f.write(f"  Median percentage: {np.median(percentages):.2f}%\n")
            f.write(f"  Standard deviation: {np.std(percentages):.2f}%\n")
            f.write(f"  Min percentage: {min(percentages):.2f}%\n")
            f.write(f"  Max percentage: {max(percentages):.2f}%\n")
        
        if scores:
            f.write(f"\nSimilarity Score Statistics:\n")
            f.write(f"  Mean score: {np.mean(scores):.4f}\n")
            f.write(f"  Median score: {np.median(scores):.4f}\n")
            f.write(f"  Standard deviation: {np.std(scores):.4f}\n")
            f.write(f"  Min score: {min(scores):.4f}\n")
            f.write(f"  Max score: {max(scores):.4f}\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write("DETAILED STATISTICS BY DOCUMENT\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by percentage (descending)
        sorted_stats = sorted(stats_dict.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        f.write(f"{'Document':<40} {'Total Chunks':<15} {'Plagiarized':<15} {'Percentage':<15} {'Matches':<10} {'Avg Score':<12}\n")
        f.write("-" * 107 + "\n")
        
        for doc_name, stats in sorted_stats:
            f.write(f"{doc_name:<40} {stats['total_chunks']:<15} {stats['plagiarized_chunks']:<15} "
                   f"{stats['percentage']:<15.2f} {stats['num_matches']:<10} {stats['avg_score']:<12.4f}\n")
    
    logger.log(f"Statistics report saved to {output_file}")


def query_single_paper(
    query_text: str,
    query_name: str,
    index: BasicInvertedIndex,
    ranker: Ranker,
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    top_k_papers: int = 5,
    top_chunks_per_paper: int = 2,
    similarity_threshold: float = 0.5,
    use_sbert: bool = False,
    sbert_ranker: Optional[object] = None,
    use_jaccard: bool = False,
    jaccard_ranker: Optional[object] = None,
    query_doc_id: int = None,  # ID of the query paper in the corpus (to exclude from results)
    sbert_threshold: float = 0.65,
    jaccard_threshold: float = 0.4
) -> Dict[str, List[Dict]]:
    """
    Query a single paper against the corpus and return top matches with best chunks.
    
    Args:
        query_text: Full text of the paper to query
        query_name: Name/identifier for the query paper
        index: The inverted index
        ranker: The ranker to use for querying
        doc_texts: Dictionary mapping doc_id to full text
        doc_id_mapping: Mapping from numeric doc_id to original filename
        logger: Logger instance
        top_k_papers: Number of top papers to return (default: 5)
        top_chunks_per_paper: Number of top chunks per paper (default: 2)
        similarity_threshold: Minimum similarity score
        use_sbert: Whether to use SBERT
        sbert_ranker: SBERTRanker instance
        use_jaccard: Whether to use Jaccard
        jaccard_ranker: JaccardRanker instance
    
    Returns:
        Dictionary with query results: {
            'query_name': str,
            'results': [
                {
                    'paper_name': str,
                    'score': float,
                    'chunks': [
                        {'chunk_idx': int, 'chunk_text': str, 'score': float, 'sbert_score': float, 'jaccard_score': float}
                    ]
                }
            ]
        }
    """
    logger.log(f"\nQuerying paper: {query_name}")
    if query_doc_id is not None:
        logger.log(f"Excluding query paper (doc_id: {query_doc_id}) from results")
    
    # Use BM25 to get candidate papers
    bm25_results = ranker.query(query_text)
    logger.log(f"BM25 returned {len(bm25_results)} candidate papers")
    
    # Get more candidates than needed to ensure we have enough after filtering
    candidate_count = max(top_k_papers * 3, 20)
    candidates = []
    for matched_doc_id, bm25_score in bm25_results[:candidate_count]:
        # Exclude the query paper itself
        if query_doc_id is not None and matched_doc_id == query_doc_id:
            continue
        matched_text = doc_texts.get(matched_doc_id, "")
        if matched_text:
            candidates.append((matched_doc_id, bm25_score, matched_text))
    
    logger.log(f"Found {len(candidates)} candidate papers (excluding query paper)")
    
    # Extract chunks from query paper
    query_chunks = extract_chunks(query_text, chunk_size=200, overlap=50)
    query_chunks = [chunk for chunk in query_chunks if len(chunk.split()) >= 20]
    logger.log(f"Extracted {len(query_chunks)} query chunks")
    
    # For each candidate, calculate similarity scores and find best chunks
    paper_results = []
    for matched_doc_id, bm25_score, matched_text in candidates:
        paper_name = doc_id_mapping.get(matched_doc_id, str(matched_doc_id))
        matched_chunks = extract_chunks(matched_text, chunk_size=200, overlap=50)
        matched_chunks = [chunk for chunk in matched_chunks if len(chunk.split()) >= 20]
        
        # Find best matching chunks and calculate scores
        best_chunks = []
        max_sbert_score = 0.0
        max_jaccard_score = 0.0
        seen_matched_chunks = {}  # Track best match for each matched chunk: {matched_chunk_idx: (query_chunk_idx, score, ...)}
        
        for query_chunk_idx, query_chunk in enumerate(query_chunks[:10]):  # Check top 10 query chunks
            for chunk_idx, matched_chunk in enumerate(matched_chunks):
                # Calculate Jaccard similarity
                query_words = set(word.lower() for word in query_chunk.split())
                matched_words = set(word.lower() for word in matched_chunk.split())
                
                if len(query_words) == 0 or len(matched_words) == 0:
                    continue
                
                intersection = len(query_words & matched_words)
                union = len(query_words | matched_words)
                jaccard_score = intersection / union if union > 0 else 0.0
                
                # Calculate SBERT score if available
                sbert_score = 0.0
                if use_sbert and sbert_ranker:
                    try:
                        _, _, sbert_score = find_best_matching_chunk(
                            query_chunk, matched_chunk, use_sbert=True, sbert_ranker=sbert_ranker,
                            sbert_threshold=sbert_threshold
                        )
                    except:
                        pass
                
                # Use the best scoring method for this chunk
                if use_sbert and sbert_ranker and sbert_score > jaccard_score:
                    chunk_score = sbert_score
                else:
                    chunk_score = jaccard_score
                
                # Track the best match for this matched chunk
                if chunk_idx not in seen_matched_chunks or chunk_score > seen_matched_chunks[chunk_idx][2]:
                    seen_matched_chunks[chunk_idx] = (
                        query_chunk_idx,
                        query_chunk,
                        chunk_score,
                        chunk_idx,
                        matched_chunk,
                        sbert_score,
                        jaccard_score
                    )
        
        # Convert to list format
        for matched_chunk_idx, (query_chunk_idx, query_chunk, chunk_score, _, matched_chunk, sbert_score, jaccard_score) in seen_matched_chunks.items():
            best_chunks.append({
                'query_chunk_idx': query_chunk_idx,
                'query_chunk_text': query_chunk[:500] + '...' if len(query_chunk) > 500 else query_chunk,
                'matched_chunk_idx': matched_chunk_idx,
                'matched_chunk_text': matched_chunk[:500] + '...' if len(matched_chunk) > 500 else matched_chunk,
                'score': chunk_score,
                'sbert_score': sbert_score,
                'jaccard_score': jaccard_score
            })
            max_sbert_score = max(max_sbert_score, sbert_score)
            max_jaccard_score = max(max_jaccard_score, jaccard_score)
        
        # Sort chunks by score and take top N
        best_chunks.sort(key=lambda x: x['score'], reverse=True)
        best_chunks = best_chunks[:top_chunks_per_paper]
        
        # Calculate overall paper score (hybrid if enabled, otherwise best similarity)
        if use_sbert and sbert_ranker:
            # Use SBERT as primary score
            paper_score = max_sbert_score if max_sbert_score > 0 else max_jaccard_score
        else:
            # Use Jaccard as primary score
            paper_score = max_jaccard_score
        
        # If no chunks found, use BM25 score as fallback
        if paper_score == 0:
            paper_score = bm25_score / 100.0  # Normalize BM25 score roughly
        
        paper_results.append({
            'paper_name': paper_name,
            'score': paper_score,
            'bm25_score': bm25_score,
            'sbert_score': max_sbert_score,
            'jaccard_score': max_jaccard_score,
            'chunks': best_chunks if best_chunks else [{
                'query_chunk_idx': 0,
                'query_chunk_text': query_chunks[0][:500] + '...' if query_chunks and len(query_chunks[0]) > 500 else (query_chunks[0] if query_chunks else 'N/A'),
                'matched_chunk_idx': 0,
                'matched_chunk_text': matched_text[:500] + '...' if len(matched_text) > 500 else matched_text,
                'score': 0.0,
                'sbert_score': 0.0,
                'jaccard_score': 0.0
            }]
        })
    
    # Sort by score and return top K (always return top_k_papers even if scores are low)
    paper_results.sort(key=lambda x: x['score'], reverse=True)
    top_results = paper_results[:top_k_papers]
    
    logger.log(f"Returning {len(top_results)} papers (top {top_k_papers} by similarity/hybrid score)")
    for i, result in enumerate(top_results, 1):
        logger.log(f"  {i}. {result['paper_name']} (score: {result['score']:.4f}, BM25: {result['bm25_score']:.2f}, chunks: {len(result['chunks'])})")
    
    return {
        'query_name': query_name,
        'results': top_results
    }


def generate_evaluation_queries(
    query_papers: Dict[str, str],
    index: BasicInvertedIndex,
    ranker: Ranker,
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    output_file: str = "evaluation_queries.json",
    top_k_papers: int = 5,
    top_chunks_per_paper: int = 2,
    similarity_threshold: float = 0.5,
    use_sbert: bool = False,
    sbert_ranker: Optional[object] = None,
    use_jaccard: bool = False,
    jaccard_ranker: Optional[object] = None,
    sbert_threshold: float = 0.65,
    jaccard_threshold: float = 0.4
) -> None:
    """
    Generate evaluation queries for a set of papers.
    
    Args:
        query_papers: Dictionary mapping paper names to their text content
        index: The inverted index
        ranker: The ranker to use
        doc_texts: Dictionary mapping doc_id to full text
        doc_id_mapping: Mapping from numeric doc_id to original filename
        logger: Logger instance
        output_file: Path to save evaluation results (JSON format)
        top_k_papers: Number of top papers to return per query
        top_chunks_per_paper: Number of top chunks per paper
        similarity_threshold: Minimum similarity score
        use_sbert: Whether to use SBERT
        sbert_ranker: SBERTRanker instance
        use_jaccard: Whether to use Jaccard
        jaccard_ranker: JaccardRanker instance
    """
    logger.log("\n" + "=" * 80)
    logger.log("GENERATING EVALUATION QUERIES")
    logger.log("=" * 80)
    logger.log(f"Processing {len(query_papers)} query papers")
    
    evaluation_results = []
    
    for query_idx, (query_name, query_text) in enumerate(query_papers.items(), 1):
        logger.log(f"\n[{query_idx}/{len(query_papers)}] Processing query: {query_name}")
        
        try:
            result = query_single_paper(
                query_text=query_text,
                query_name=query_name,
                index=index,
                ranker=ranker,
                doc_texts=doc_texts,
                doc_id_mapping=doc_id_mapping,
                logger=logger,
                top_k_papers=top_k_papers,
                top_chunks_per_paper=top_chunks_per_paper,
                similarity_threshold=similarity_threshold,
                use_sbert=use_sbert,
                sbert_ranker=sbert_ranker,
                use_jaccard=use_jaccard,
                jaccard_ranker=jaccard_ranker,
                sbert_threshold=sbert_threshold,
                jaccard_threshold=jaccard_threshold
            )
            evaluation_results.append(result)
        except Exception as e:
            logger.log_error(f"Failed to process query {query_name}", e)
            continue
    
    # Save results to JSON file
    logger.log(f"\nSaving evaluation results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.log(f"Saved {len(evaluation_results)} evaluation queries")
    
    # Also generate a human-readable report
    report_file = output_file.replace('.json', '_report.txt')
    logger.log(f"Generating human-readable report: {report_file}")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION QUERIES REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for query_result in evaluation_results:
            f.write("-" * 80 + "\n")
            f.write(f"QUERY PAPER: {query_result['query_name']}\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Top {top_k_papers} Matching Papers:\n\n")
            
            for rank, result in enumerate(query_result['results'], 1):
                f.write(f"{rank}. {result['paper_name']}\n")
                f.write(f"   Overall Score: {result['score']:.4f}\n")
                f.write(f"   Top {top_chunks_per_paper} Matching Chunks:\n\n")
                
                for chunk_idx, chunk_info in enumerate(result['chunks'], 1):
                    f.write(f"   Chunk Pair {chunk_idx}:\n")
                    f.write(f"   - Similarity Score: {chunk_info['score']:.4f}\n")
                    if chunk_info.get('sbert_score', 0) > 0:
                        f.write(f"   - SBERT Score: {chunk_info['sbert_score']:.4f}\n")
                    if chunk_info.get('jaccard_score', 0) > 0:
                        f.write(f"   - Jaccard Score: {chunk_info['jaccard_score']:.4f}\n")
                    
                    # Handle both old and new format
                    query_chunk_text = chunk_info.get('query_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                    matched_chunk_text = chunk_info.get('matched_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                    query_chunk_idx = chunk_info.get('query_chunk_idx', chunk_info.get('chunk_idx', '?'))
                    matched_chunk_idx = chunk_info.get('matched_chunk_idx', chunk_info.get('chunk_idx', '?'))
                    
                    f.write(f"\n   Query Chunk #{query_chunk_idx}:\n")
                    f.write(f"   {query_chunk_text}\n\n")
                    f.write(f"   Matched Chunk #{matched_chunk_idx}:\n")
                    f.write(f"   {matched_chunk_text}\n\n")
                
                f.write("\n")
            
            f.write("\n")
    
    logger.log(f"Evaluation complete. Results saved to {output_file} and {report_file}")


def main(mode: str = "detection", query_paper_path: str = None, query_paper_name: str = None, 
         evaluation_papers: Dict[str, str] = None, auto_query_count: int = 5):
    """
    Main function to run plagiarism detection pipeline.
    
    Args:
        mode: Operation mode - "detection" (default), "query", "evaluation", or "auto_query"
        query_paper_path: Path to a paper file to query (for query mode)
        query_paper_name: Name for the query paper (for query mode)
        evaluation_papers: Dictionary of {paper_name: paper_text} for evaluation mode
        auto_query_count: Number of papers to automatically query (for auto_query mode)
    """
    # Configuration
    corpus_folder = "corpus_arxiv_Edward_Witten"  # Corpus folder with metadata
    stopwords_file = "stopwords.txt"
    index_cache_dir = "plagiarism_index_cache"
    log_file = "plagiarism_detection.log"
    similarity_threshold = 0.5  # BM25 threshold (increased from 0.3 for stricter detection)
    sbert_threshold = 0.65  # SBERT threshold (stricter for semantic similarity)
    jaccard_threshold = 0.4  # Jaccard threshold (if using Jaccard)
    min_word_overlap_ratio = 0.3  # Minimum word overlap ratio between chunks
    top_k = 5  # Number of top matches to consider per chunk
    use_sbert = True  # Set to True to enable SBERT semantic similarity
    use_jaccard = False  # Set to True to enable Jaccard similarity
    hybrid_mode = False  # Set to True to combine multiple ranking methods
    sbert_model = "all-MiniLM-L6-v2"  # SBERT model to use
    bm25_weight = 0.3  # Weight for BM25 scores in hybrid mode
    sbert_weight = 0.4  # Weight for SBERT scores in hybrid mode
    jaccard_weight = 0.3  # Weight for Jaccard scores in hybrid mode
    
    # Query/Evaluation specific settings
    top_k_papers = 5  # Number of top papers to return
    top_chunks_per_paper = 2  # Number of top chunks per paper
    
    # Initialize logger
    logger = Logger(log_file)
    
    try:
        logger.log("=" * 80)
        logger.log("PLAGIARISM DETECTION SYSTEM")
        logger.log("=" * 80)
        
        # Step 1: Load documents from metadata
        logger.log("\nStep 1: Loading documents from metadata...")
        logger.log(f"Corpus folder: {corpus_folder}")
        try:
            doc_texts, doc_id_mapping = load_documents_from_metadata(corpus_folder, logger)
        except Exception as e:
            logger.log_error("Failed to load documents", e)
            raise
        
        if not doc_texts:
            logger.log("No documents found. Exiting.")
            return
        
        # Step 2: Create temporary JSONL file for indexing
        logger.log("\nStep 2: Preparing documents for indexing...")
        temp_jsonl = None
        temp_jsonl_path = None
        try:
            temp_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
            temp_jsonl_path = temp_jsonl.name
            temp_jsonl.close()
            
            create_jsonl_from_documents(doc_texts, temp_jsonl_path, logger)
            logger.log(f"Created temporary JSONL file: {temp_jsonl_path}")
        except Exception as e:
            logger.log_error("Failed to create JSONL file", e)
            raise
        
        # Step 3: Initialize tokenizer and stopwords
        logger.log("\nStep 3: Initializing tokenizer and loading stopwords...")
        try:
            tokenizer = RegexTokenizer(r'\w+', lowercase=True)
            
            # Load stopwords (from file if available, otherwise from NLTK)
            stopwords = load_stopwords(stopwords_file, logger, use_nltk=True)
            logger.log(f"Initialized tokenizer with {len(stopwords)} stopwords")
        except Exception as e:
            logger.log_error("Failed to initialize tokenizer or load stopwords", e)
            raise
        
        # Step 4: Create or load index
        logger.log("\nStep 4: Building index...")
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
        
        # Step 5: Initialize rankers
        logger.log("\nStep 5: Initializing rankers...")
        try:
            # Initialize BM25 ranker
            scorer = BM25(index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8})
            ranker = Ranker(
                index=index,
                document_preprocessor=tokenizer,
                stopwords=stopwords,
                scorer=scorer,
                raw_text_dict=doc_texts
            )
            logger.log("BM25 ranker initialized successfully")
            
            # Initialize SBERT ranker if requested
            sbert_ranker = None
            if use_sbert and SBERT_AVAILABLE:
                try:
                    logger.log(f"Initializing SBERT ranker (model: {sbert_model})...")
                    sbert_ranker = SBERTRanker(model_name=sbert_model)
                    logger.log("SBERT ranker initialized successfully")
                except Exception as e:
                    logger.log_error(f"Failed to initialize SBERT ranker: {e}")
                    logger.log("Falling back to BM25 only")
                    use_sbert = False
                    sbert_ranker = None
            elif use_sbert and not SBERT_AVAILABLE:
                logger.log("SBERT requested but not available. Install sentence-transformers to enable.")
                use_sbert = False
            
            # Initialize Jaccard ranker if requested
            jaccard_ranker = None
            if use_jaccard:
                if JACCARD_AVAILABLE:
                    try:
                        logger.log("Initializing Jaccard ranker...")
                        jaccard_ranker = JaccardRanker()
                        logger.log("Jaccard ranker initialized successfully")
                    except Exception as e:
                        logger.log_error(f"Failed to initialize Jaccard ranker: {e}")
                        use_jaccard = False
                else:
                    logger.log("Jaccard ranker not available.")
                    use_jaccard = False
        except Exception as e:
            logger.log_error("Failed to initialize ranker", e)
            raise
        
        # Step 6: Run based on mode
        if mode == "auto_query":
            # Auto-query mode: Automatically query N papers from the corpus
            logger.log("\nStep 6: Auto-query mode - Selecting papers to query...")
            
            # Select papers evenly distributed across the corpus
            all_doc_ids = sorted(doc_texts.keys())
            total_docs = len(all_doc_ids)
            query_count = min(auto_query_count, total_docs)
            
            # Select papers evenly spaced across the corpus
            if query_count > 0:
                step = max(1, total_docs // query_count)
                selected_indices = [i * step for i in range(query_count)]
                selected_doc_ids = [all_doc_ids[i] for i in selected_indices if i < total_docs]
                
                # If we don't have enough, add more from the end
                while len(selected_doc_ids) < query_count and len(selected_doc_ids) < total_docs:
                    for doc_id in reversed(all_doc_ids):
                        if doc_id not in selected_doc_ids:
                            selected_doc_ids.append(doc_id)
                            break
                    if len(selected_doc_ids) >= query_count:
                        break
                
                selected_doc_ids = selected_doc_ids[:query_count]
                
                logger.log(f"Selected {len(selected_doc_ids)} papers to query:")
                for i, doc_id in enumerate(selected_doc_ids, 1):
                    paper_name = doc_id_mapping.get(doc_id, str(doc_id))
                    logger.log(f"  {i}. {paper_name}")
                
                # Query each selected paper
                for query_idx, doc_id in enumerate(selected_doc_ids, 1):
                    query_name = doc_id_mapping.get(doc_id, str(doc_id))
                    query_text = doc_texts[doc_id]
                    
                    logger.log(f"\n[{query_idx}/{len(selected_doc_ids)}] Querying: {query_name}")
                    
                    try:
                        result = query_single_paper(
                            query_text=query_text,
                            query_name=query_name,
                            index=index,
                            ranker=ranker,
                            doc_texts=doc_texts,
                            doc_id_mapping=doc_id_mapping,
                            logger=logger,
                            top_k_papers=top_k_papers,
                            top_chunks_per_paper=top_chunks_per_paper,
                            similarity_threshold=similarity_threshold,
                            use_sbert=use_sbert,
                            sbert_ranker=sbert_ranker,
                            use_jaccard=use_jaccard,
                            jaccard_ranker=jaccard_ranker,
                            query_doc_id=doc_id,  # Exclude the query paper itself
                            sbert_threshold=sbert_threshold,
                            jaccard_threshold=jaccard_threshold
                        )
                        
                        # Create output directory named after the query paper
                        output_dir = Path(f"query_results_{query_name}")
                        output_dir.mkdir(exist_ok=True)
                        
                        # Save results
                        output_file = output_dir / "query_results.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                        
                        # Also save a human-readable report
                        report_file = output_dir / "query_report.txt"
                        with open(report_file, 'w', encoding='utf-8') as f:
                            f.write("=" * 80 + "\n")
                            f.write(f"QUERY RESULTS: {query_name}\n")
                            f.write("=" * 80 + "\n\n")
                            f.write(f"Top {top_k_papers} Matching Papers:\n\n")
                            
                            for rank, res in enumerate(result['results'], 1):
                                f.write(f"{rank}. {res['paper_name']}\n")
                                f.write(f"   Overall Score: {res['score']:.4f}\n")
                                f.write(f"   Top {top_chunks_per_paper} Matching Chunks:\n\n")
                                
                                for chunk_idx, chunk_info in enumerate(res['chunks'], 1):
                                    f.write(f"   Chunk {chunk_idx}:\n")
                                    f.write(f"   - Similarity Score: {chunk_info['score']:.4f}\n")
                                    if chunk_info.get('sbert_score', 0) > 0:
                                        f.write(f"   - SBERT Score: {chunk_info['sbert_score']:.4f}\n")
                                    if chunk_info.get('jaccard_score', 0) > 0:
                                        f.write(f"   - Jaccard Score: {chunk_info['jaccard_score']:.4f}\n")
                                    
                                    # Handle both old and new format
                                    query_chunk_text = chunk_info.get('query_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                                    matched_chunk_text = chunk_info.get('matched_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                                    query_chunk_idx = chunk_info.get('query_chunk_idx', chunk_info.get('chunk_idx', '?'))
                                    matched_chunk_idx = chunk_info.get('matched_chunk_idx', chunk_info.get('chunk_idx', '?'))
                                    
                                    f.write(f"\n   Query Chunk #{query_chunk_idx}:\n")
                                    f.write(f"   {query_chunk_text}\n\n")
                                    f.write(f"   Matched Chunk #{matched_chunk_idx}:\n")
                                    f.write(f"   {matched_chunk_text}\n\n")
                                
                                f.write("\n")
                        
                        logger.log(f"  Results saved to {output_dir}/")
                        
                    except Exception as e:
                        logger.log_error(f"Failed to query {query_name}", e)
                        continue
                
                logger.log(f"\nAuto-query complete. Processed {len(selected_doc_ids)} papers.")
            else:
                logger.log("No papers available to query.")
                
        elif mode == "query":
            # Query mode: Query a single paper
            if not query_paper_path:
                logger.log_error("Query mode requires query_paper_path", ValueError("query_paper_path is required for query mode"))
                raise ValueError("query_paper_path is required for query mode")
            
            # Load query paper
            logger.log(f"\nStep 6: Loading query paper from {query_paper_path}...")
            try:
                with open(query_paper_path, 'r', encoding='utf-8') as f:
                    query_text = f.read()
                query_name = query_paper_name or Path(query_paper_path).stem
                logger.log(f"Loaded query paper: {query_name}")
            except Exception as e:
                logger.log_error(f"Failed to load query paper from {query_paper_path}", e)
                raise
            
            # Query the paper
            logger.log("\nStep 7: Querying paper against corpus...")
            
            # Find the doc_id if this paper is in the corpus (to exclude it)
            query_doc_id = None
            for doc_id, name in doc_id_mapping.items():
                if name == query_name:
                    query_doc_id = doc_id
                    break
            
            try:
                result = query_single_paper(
                    query_text=query_text,
                    query_name=query_name,
                    index=index,
                    ranker=ranker,
                    doc_texts=doc_texts,
                    doc_id_mapping=doc_id_mapping,
                    logger=logger,
                    top_k_papers=top_k_papers,
                    top_chunks_per_paper=top_chunks_per_paper,
                    similarity_threshold=similarity_threshold,
                    use_sbert=use_sbert,
                    sbert_ranker=sbert_ranker,
                    use_jaccard=use_jaccard,
                    jaccard_ranker=jaccard_ranker,
                    query_doc_id=query_doc_id,  # Exclude if found in corpus
                    sbert_threshold=sbert_threshold,
                    jaccard_threshold=jaccard_threshold
                )
                
                # Create output directory named after the query paper
                output_dir = Path(f"query_results_{query_name}")
                output_dir.mkdir(exist_ok=True)
                
                # Save results
                output_file = output_dir / "query_results.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.log(f"Query results saved to {output_file}")
                
                # Also save a human-readable report
                report_file = output_dir / "query_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write(f"QUERY RESULTS: {query_name}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"Top {top_k_papers} Matching Papers:\n\n")
                    
                    for rank, res in enumerate(result['results'], 1):
                        f.write(f"{rank}. {res['paper_name']}\n")
                        f.write(f"   Overall Score: {res['score']:.4f}\n")
                        f.write(f"   Top {top_chunks_per_paper} Matching Chunks:\n\n")
                        
                        for chunk_idx, chunk_info in enumerate(res['chunks'], 1):
                            f.write(f"   Chunk Pair {chunk_idx}:\n")
                            f.write(f"   - Similarity Score: {chunk_info['score']:.4f}\n")
                            if chunk_info.get('sbert_score', 0) > 0:
                                f.write(f"   - SBERT Score: {chunk_info['sbert_score']:.4f}\n")
                            if chunk_info.get('jaccard_score', 0) > 0:
                                f.write(f"   - Jaccard Score: {chunk_info['jaccard_score']:.4f}\n")
                            
                            # Handle both old and new format
                            query_chunk_text = chunk_info.get('query_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                            matched_chunk_text = chunk_info.get('matched_chunk_text') or chunk_info.get('chunk_text', 'N/A')
                            query_chunk_idx = chunk_info.get('query_chunk_idx', chunk_info.get('chunk_idx', '?'))
                            matched_chunk_idx = chunk_info.get('matched_chunk_idx', chunk_info.get('chunk_idx', '?'))
                            
                            f.write(f"\n   Query Chunk #{query_chunk_idx}:\n")
                            f.write(f"   {query_chunk_text}\n\n")
                            f.write(f"   Matched Chunk #{matched_chunk_idx}:\n")
                            f.write(f"   {matched_chunk_text}\n\n")
                        
                        f.write("\n")
                
                logger.log(f"Query report saved to {report_file}")
                logger.log(f"All query results saved to directory: {output_dir}")
                
            except Exception as e:
                logger.log_error("Failed during query", e)
                raise
                
        elif mode == "evaluation":
            # Evaluation mode: Generate evaluation queries
            if not evaluation_papers:
                logger.log_error("Evaluation mode requires evaluation_papers", ValueError("evaluation_papers is required for evaluation mode"))
                raise ValueError("evaluation_papers is required for evaluation mode")
            
            logger.log("\nStep 6: Generating evaluation queries...")
            try:
                generate_evaluation_queries(
                    query_papers=evaluation_papers,
                    index=index,
                    ranker=ranker,
                    doc_texts=doc_texts,
                    doc_id_mapping=doc_id_mapping,
                    logger=logger,
                    output_file="evaluation_queries.json",
                    top_k_papers=top_k_papers,
                    top_chunks_per_paper=top_chunks_per_paper,
                    similarity_threshold=similarity_threshold,
                    use_sbert=use_sbert,
                    sbert_ranker=sbert_ranker,
                    use_jaccard=use_jaccard,
                    jaccard_ranker=jaccard_ranker,
                    sbert_threshold=sbert_threshold,
                    jaccard_threshold=jaccard_threshold
                )
            except Exception as e:
                logger.log_error("Failed during evaluation generation", e)
                raise
                
        else:
            # Default: Detection mode
            logger.log("\nStep 6: Running plagiarism detection...")
            try:
                plagiarism_results = detect_plagiarism(
                    index=index,
                    ranker=ranker,
                    doc_texts=doc_texts,
                    doc_id_mapping=doc_id_mapping,
                    logger=logger,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k,
                    use_sbert=use_sbert,
                    sbert_ranker=sbert_ranker,
                    use_jaccard=use_jaccard,
                    jaccard_ranker=jaccard_ranker,
                    hybrid_mode=hybrid_mode,
                    bm25_weight=bm25_weight,
                    sbert_weight=sbert_weight,
                    jaccard_weight=jaccard_weight,
                    sbert_threshold=sbert_threshold,
                    jaccard_threshold=jaccard_threshold,
                    min_word_overlap_ratio=min_word_overlap_ratio
                )
            except Exception as e:
                logger.log_error("Failed during plagiarism detection", e)
                raise
            
            # Step 7: Create results directory
            results_dir = Path("results_edward_witten")
            results_dir.mkdir(exist_ok=True)
            logger.log(f"\nStep 7: Saving results to {results_dir}/...")
            
            # Step 8: Save plagiarism chunks dictionary
            logger.log("\nStep 8: Saving plagiarism chunks dictionary...")
            try:
                chunks_dict_file = results_dir / "plagiarism_chunks.json"
                save_plagiarism_chunks_dict(plagiarism_results, str(chunks_dict_file), logger)
            except Exception as e:
                logger.log_error("Failed to save plagiarism chunks dictionary", e)
                # Continue even if this fails
            
            # Step 9: Calculate statistics
            logger.log("\nStep 9: Calculating statistics...")
            try:
                stats_dict = calculate_statistics(
                    plagiarism_results,
                    doc_texts,
                    doc_id_mapping,
                    logger
                )
                
                # Save statistics as JSON
                stats_json_file = results_dir / "plagiarism_statistics.json"
                with open(stats_json_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_dict, f, indent=2, ensure_ascii=False)
                logger.log(f"Statistics saved to {stats_json_file}")
                
                # Save statistics report
                stats_report_file = results_dir / "plagiarism_statistics_report.txt"
                save_statistics_report(stats_dict, str(stats_report_file), logger)
            except Exception as e:
                logger.log_error("Failed to calculate statistics", e)
                stats_dict = {}
            
            # Step 10: Create visualizations
            logger.log("\nStep 10: Creating visualizations...")
            try:
                if stats_dict:
                    create_visualizations(stats_dict, results_dir, logger)
            except Exception as e:
                logger.log_error("Failed to create visualizations", e)
                # Continue even if visualizations fail
            
            # Step 11: Generate report with yes/no and examples
            logger.log("\nStep 11: Generating report with yes/no answers and examples...")
            try:
                report_file = results_dir / "plagiarism_report.txt"
                all_doc_names = [doc_id_mapping[doc_id] for doc_id in doc_texts.keys()]
                generate_report(plagiarism_results, str(report_file), logger, 
                              similarity_threshold=similarity_threshold, all_doc_names=all_doc_names)
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
        if mode == "query":
            logger.log("QUERY COMPLETE")
        elif mode == "auto_query":
            logger.log("AUTO-QUERY COMPLETE")
        elif mode == "evaluation":
            logger.log("EVALUATION GENERATION COMPLETE")
        else:
            logger.log("PLAGIARISM DETECTION COMPLETE")
        logger.log("=" * 80)
        
        # Print summary statistics
        if mode == "detection":
            total_docs_with_matches = len([d for d, matches in plagiarism_results.items() if matches])
            total_matches = sum(len(matches) for matches in plagiarism_results.values())
            
            logger.log(f"\nSummary:")
            logger.log(f"  Total documents analyzed: {len(doc_texts)}")
            logger.log(f"  Documents with plagiarism detected (YES): {total_docs_with_matches}")
            logger.log(f"  Documents with no plagiarism detected (NO): {len(doc_texts) - total_docs_with_matches}")
            logger.log(f"  Total potential matches found: {total_matches}")
            logger.log(f"  Threshold: {similarity_threshold} (score > {similarity_threshold} = YES, otherwise NO)")
            logger.log(f"\nAll results saved to: results_edward_witten/")
            logger.log(f"  - Report: plagiarism_report.txt")
            logger.log(f"  - Chunks dictionary: plagiarism_chunks.json")
            logger.log(f"  - Statistics: plagiarism_statistics.json and plagiarism_statistics_report.txt")
            logger.log(f"  - Visualizations: *.png files")
        elif mode == "query":
            logger.log(f"\nQuery complete. Results saved to query_results_*/ directory")
        elif mode == "auto_query":
            logger.log(f"\nAuto-query complete. Results saved to query_results_*/ directories")
        elif mode == "evaluation":
            logger.log(f"\nEvaluation complete. Results saved to evaluation_queries.json and evaluation_queries_report.txt")
        
        logger.log(f"Log file saved to: {log_file}")
        
    except Exception as e:
        logger.log_error("Fatal error in main execution", e)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plagiarism Detection System')
    parser.add_argument('--mode', type=str, default='detection', 
                       choices=['detection', 'query', 'evaluation', 'auto_query'],
                       help='Operation mode: detection (default), query, evaluation, or auto_query')
    parser.add_argument('--query-paper', type=str, 
                       help='Path to paper file to query (for query mode)')
    parser.add_argument('--query-name', type=str,
                       help='Name for the query paper (optional, uses filename if not provided)')
    parser.add_argument('--evaluation-papers-dir', type=str,
                       help='Directory containing papers for evaluation (for evaluation mode)')
    parser.add_argument('--auto-query-count', type=int, default=5,
                       help='Number of papers to automatically query (for auto_query mode, default: 5)')
    
    args = parser.parse_args()
    
    # Handle evaluation mode: load papers from directory
    evaluation_papers = None
    if args.mode == "evaluation":
        if args.evaluation_papers_dir:
            evaluation_papers = {}
            eval_dir = Path(args.evaluation_papers_dir)
            if eval_dir.exists():
                for txt_file in eval_dir.glob("*.txt"):
                    try:
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            evaluation_papers[txt_file.stem] = f.read()
                    except Exception as e:
                        print(f"Warning: Failed to load {txt_file}: {e}")
                print(f"Loaded {len(evaluation_papers)} papers for evaluation")
            else:
                print(f"Error: Evaluation directory not found: {args.evaluation_papers_dir}")
                exit(1)
        else:
            print("Error: --evaluation-papers-dir is required for evaluation mode")
            exit(1)
    
    main(
        mode=args.mode,
        query_paper_path=args.query_paper,
        query_paper_name=args.query_name,
        evaluation_papers=evaluation_papers,
        auto_query_count=args.auto_query_count
    )
