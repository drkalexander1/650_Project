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


def detect_plagiarism(
    index: BasicInvertedIndex,
    ranker: Ranker,
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    similarity_threshold: float = 0.3,
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float, str]]]:
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
        Dictionary mapping document ID to list of (matched_doc_id, score, matched_chunk) tuples
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
                        (matched_doc_id, score, chunk[:200])  # Store first 200 chars of chunk
                        for matched_doc_id, score in results
                        if matched_doc_id != doc_id and score >= similarity_threshold
                    ][:top_k]
                    
                    if filtered_results:
                        matches_found += len(filtered_results)
                        for matched_doc_id, score, matched_chunk in filtered_results:
                            matched_doc_name = doc_id_mapping.get(matched_doc_id, str(matched_doc_id))
                            plagiarism_results[doc_name].append((
                                matched_doc_name,
                                score,
                                matched_chunk
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


def generate_report(plagiarism_results: Dict[str, List[Tuple[str, float, str]]], 
                   output_file: str = "plagiarism_report.txt",
                   logger: Logger = None) -> None:
    """
    Generate a human-readable plagiarism detection report.
    
    Args:
        plagiarism_results: Results from detect_plagiarism
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
            for matched_doc, score, chunk in matches:
                matches_by_doc[matched_doc].append((score, chunk))
            
            for matched_doc, doc_matches in sorted(
                matches_by_doc.items(),
                key=lambda x: max(score for score, _ in x[1]),
                reverse=True
            ):
                f.write(f"  Matches with: {matched_doc}\n")
                f.write(f"  Number of similar sections: {len(doc_matches)}\n")
                f.write(f"  Highest similarity score: {max(score for score, _ in doc_matches):.4f}\n")
                f.write(f"  Average similarity score: {sum(score for score, _ in doc_matches) / len(doc_matches):.4f}\n")
                f.write(f"  Sample matched text: {doc_matches[0][1][:150]}...\n\n")
        
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
        logger.log("\nStep 3: Initializing tokenizer...")
        try:
            tokenizer = RegexTokenizer(r'\w+', lowercase=True)
            
            # Simple stopwords set (you can expand this)
            stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
                'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
            }
            logger.log(f"Initialized tokenizer with {len(stopwords)} stopwords")
        except Exception as e:
            logger.log_error("Failed to initialize tokenizer", e)
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

