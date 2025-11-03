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

from preprocessing import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import Ranker, BM25


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


def load_stopwords(stopwords_file: str, logger: Logger) -> set:
    """
    Load stopwords from a text file.
    
    Args:
        stopwords_file: Path to the stopwords file (space-separated words)
        logger: Logger instance for output
        
    Returns:
        Set of stopwords
    """
    stopwords_path = Path(stopwords_file)
    
    if not stopwords_path.exists():
        logger.log_error(f"Stopwords file not found: {stopwords_file}")
        raise FileNotFoundError(f"Stopwords file not found: {stopwords_file}")
    
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Split by whitespace and convert to set
            stopwords = set(word.strip().lower() for word in content.split() if word.strip())
            logger.log(f"Loaded {len(stopwords)} stopwords from {stopwords_file}")
            return stopwords
    except Exception as e:
        logger.log_error(f"Error reading stopwords file {stopwords_file}", e)
        raise


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


def find_best_matching_chunk(source_chunk: str, target_text: str, chunk_size: int = 200, overlap: int = 50) -> Tuple[int, str]:
    """
    Find the chunk in target_text that best matches source_chunk.
    
    Args:
        source_chunk: The source chunk text from paper A
        target_text: The full text of the target document (paper B)
        chunk_size: Size of chunks to extract
        overlap: Overlap between chunks
        
    Returns:
        Tuple of (best_chunk_idx, best_chunk_text)
    """
    source_words = set(word.lower() for word in source_chunk.split())
    target_chunks = extract_chunks(target_text, chunk_size=chunk_size, overlap=overlap)
    
    best_match_idx = 0
    best_match_score = 0
    best_match_text = ""
    
    for idx, target_chunk in enumerate(target_chunks):
        if len(target_chunk.split()) < 20:
            continue
        
        target_words = set(word.lower() for word in target_chunk.split())
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(source_words & target_words)
        union = len(source_words | target_words)
        similarity = intersection / union if union > 0 else 0
        
        if similarity > best_match_score:
            best_match_score = similarity
            best_match_idx = idx
            best_match_text = target_chunk
    
    return best_match_idx, best_match_text


def detect_plagiarism(
    index: BasicInvertedIndex,
    ranker: Ranker,
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    similarity_threshold: float = 0.3,
    top_k: int = 10
) -> Dict[str, List[Tuple[str, int, str, float, int, str]]]:
    """
    Detect plagiarism by querying chunks from each document against the index.
    
    Args:
        index: The inverted index
        ranker: The ranker to use for querying
        doc_texts: Dictionary mapping doc_id to full text
        doc_id_mapping: Mapping from numeric doc_id to original filename
        logger: Logger instance for output
        similarity_threshold: Minimum similarity score to consider as plagiarism
        top_k: Number of top results to consider per chunk
        
    Returns:
        Dictionary mapping document ID to list of tuples:
        (matched_doc_name, source_chunk_idx, source_chunk_text, score, matched_chunk_idx, matched_chunk_text)
    """
    plagiarism_results = defaultdict(list)
    
    total_docs = len(doc_texts)
    logger.log(f"Detecting plagiarism across {total_docs} documents...")
    logger.log(f"Similarity threshold: {similarity_threshold}, Top-K: {top_k}")
    
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
                    # Query this chunk
                    results = ranker.query(chunk)
                    
                    # Filter results: exclude the document itself and low-scoring matches
                    filtered_results = [
                        (matched_doc_id, score)
                        for matched_doc_id, score in results
                        if matched_doc_id != doc_id and score >= similarity_threshold
                    ][:top_k]
                    
                    if filtered_results:
                        matches_found += len(filtered_results)
                        for matched_doc_id, score in filtered_results:
                            matched_doc_name = doc_id_mapping.get(matched_doc_id, str(matched_doc_id))
                            
                            # Find the best matching chunk in the matched document
                            matched_text = doc_texts.get(matched_doc_id, "")
                            if matched_text:
                                matched_chunk_idx, matched_chunk_text = find_best_matching_chunk(
                                    chunk, matched_text
                                )
                                
                                plagiarism_results[doc_name].append((
                                    matched_doc_name,
                                    chunk_idx,  # Source chunk index from paper A
                                    chunk,      # Source chunk text from paper A
                                    score,      # BM25 similarity score
                                    matched_chunk_idx,  # Matched chunk index from paper B
                                    matched_chunk_text  # Matched chunk text from paper B
                                ))
                except Exception as e:
                    logger.log_error(f"Error querying chunk {chunk_idx} from {doc_name}", e)
                    continue
            
            logger.log(f"  Processed {chunks_processed} chunks, found {matches_found} potential matches")
            
        except Exception as e:
            logger.log_error(f"Error processing document {doc_name}", e)
            continue
    
    logger.log(f"\nPlagiarism detection complete. Found matches in {len(plagiarism_results)} documents")
    return dict(plagiarism_results)


def generate_report(plagiarism_results: Dict[str, List[Tuple[str, int, str, float, int, str]]], 
                   output_file: str = "plagiarism_report.txt",
                   logger: Logger = None) -> None:
    """
    Generate a human-readable plagiarism detection report with specific chunk pairs.
    
    Args:
        plagiarism_results: Results from detect_plagiarism
            Each match is a tuple: (matched_doc_name, source_chunk_idx, source_chunk_text, 
                                   score, matched_chunk_idx, matched_chunk_text)
        output_file: Path to output report file
        logger: Logger instance for output
    """
    if logger:
        logger.log(f"Generating plagiarism report: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PLAGIARISM DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        total_matches = sum(len(matches) for matches in plagiarism_results.values())
        f.write(f"Total documents analyzed: {len(plagiarism_results)}\n")
        f.write(f"Total potential plagiarism matches: {total_matches}\n\n")
        
        # Sort documents by number of matches (descending)
        sorted_docs = sorted(
            plagiarism_results.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for doc_name, matches in sorted_docs:
            if not matches:
                continue
                
            f.write("-" * 80 + "\n")
            f.write(f"Document: {doc_name}\n")
            f.write(f"Number of potential plagiarism matches: {len(matches)}\n")
            f.write("-" * 80 + "\n\n")
            
            # Group matches by matched document
            matches_by_doc = defaultdict(list)
            for matched_doc, source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk in matches:
                matches_by_doc[matched_doc].append((
                    source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk
                ))
            
            for matched_doc, doc_matches in sorted(
                matches_by_doc.items(),
                key=lambda x: max(score for _, _, score, _, _ in x[1]),
                reverse=True
            ):
                f.write(f"  Matches with: {matched_doc}\n")
                f.write(f"  Number of similar sections: {len(doc_matches)}\n")
                f.write(f"  Highest similarity score: {max(score for _, _, score, _, _ in doc_matches):.4f}\n")
                f.write(f"  Average similarity score: {sum(score for _, _, score, _, _ in doc_matches) / len(doc_matches):.4f}\n")
                f.write("\n")
                
                # Display chunk pairs, sorted by similarity score (highest first)
                sorted_chunk_matches = sorted(
                    doc_matches,
                    key=lambda x: x[2],  # Sort by score
                    reverse=True
                )
                
                # Limit to top 20 chunk pairs per document pair to keep report manageable
                for idx, (source_chunk_idx, source_chunk, score, matched_chunk_idx, matched_chunk) in enumerate(sorted_chunk_matches[:20]):
                    f.write(f"    --- Chunk Pair #{idx + 1} (BM25 Score: {score:.4f}) ---\n")
                    f.write(f"    Source Document: {doc_name}, Chunk #{source_chunk_idx}\n")
                    f.write(f"    Matched Document: {matched_doc}, Chunk #{matched_chunk_idx}\n")
                    f.write(f"\n    Chunk from {doc_name} (Chunk #{source_chunk_idx}):\n")
                    f.write(f"    {source_chunk[:500]}{'...' if len(source_chunk) > 500 else ''}\n")
                    f.write(f"\n    Matched chunk from {matched_doc} (Chunk #{matched_chunk_idx}):\n")
                    f.write(f"    {matched_chunk[:500]}{'...' if len(matched_chunk) > 500 else ''}\n")
                    f.write("\n")
                
                if len(sorted_chunk_matches) > 20:
                    f.write(f"    ... and {len(sorted_chunk_matches) - 20} more chunk pairs (showing top 20)\n\n")
                
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
    Main function to run plagiarism detection pipeline.
    """
    # Configuration
    text_folder = "corpus/text"
    stopwords_file = "stopwords.txt"
    index_cache_dir = "plagiarism_index_cache"
    log_file = "plagiarism_detection.log"
    similarity_threshold = 0.3  # Adjust based on your needs
    top_k = 5  # Number of top matches to consider per chunk
    
    # Initialize logger
    logger = Logger(log_file)
    
    try:
        logger.log("=" * 80)
        logger.log("PLAGIARISM DETECTION SYSTEM")
        logger.log("=" * 80)
        
        # Step 1: Load documents from text folder
        logger.log("\nStep 1: Loading documents from text folder...")
        try:
            doc_texts, doc_id_mapping = load_documents_from_text_folder(text_folder, logger)
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
            
            # Load stopwords from file
            stopwords = load_stopwords(stopwords_file, logger)
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
        
        # Step 5: Initialize ranker
        logger.log("\nStep 5: Initializing ranker...")
        try:
            scorer = BM25(index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8})
            ranker = Ranker(
                index=index,
                document_preprocessor=tokenizer,
                stopwords=stopwords,
                scorer=scorer,
                raw_text_dict=doc_texts
            )
            logger.log("Ranker initialized successfully (BM25)")
        except Exception as e:
            logger.log_error("Failed to initialize ranker", e)
            raise
        
        # Step 6: Detect plagiarism
        logger.log("\nStep 6: Running plagiarism detection...")
        try:
            plagiarism_results = detect_plagiarism(
                index=index,
                ranker=ranker,
                doc_texts=doc_texts,
                doc_id_mapping=doc_id_mapping,
                logger=logger,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
        except Exception as e:
            logger.log_error("Failed during plagiarism detection", e)
            raise
        
        # Step 7: Generate report
        logger.log("\nStep 7: Generating report...")
        try:
            generate_report(plagiarism_results, "plagiarism_report.txt", logger)
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
        
        # Print summary statistics
        total_docs_with_matches = len([d for d, matches in plagiarism_results.items() if matches])
        total_matches = sum(len(matches) for matches in plagiarism_results.values())
        
        logger.log(f"\nSummary:")
        logger.log(f"  Total documents analyzed: {len(doc_texts)}")
        logger.log(f"  Documents with potential plagiarism: {total_docs_with_matches}")
        logger.log(f"  Total potential matches found: {total_matches}")
        logger.log(f"\nDetailed report saved to: plagiarism_report.txt")
        logger.log(f"Log file saved to: {log_file}")
        
    except Exception as e:
        logger.log_error("Fatal error in main execution", e)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()

