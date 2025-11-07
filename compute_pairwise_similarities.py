#!/usr/bin/env python3
"""
Compute pairwise similarities for all papers in the corpus.
Generates comprehensive heatmaps showing all paper-to-paper comparisons.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys

# Add the project directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import RegexTokenizer, load_nltk_stopwords
from indexing import BasicInvertedIndex, Indexer, IndexType
from ranker import Ranker, BM25
import tempfile

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


class Logger:
    """Simple logger for output."""
    def log(self, message: str):
        print(message, flush=True)
    
    def log_error(self, message: str, error: Exception = None):
        print(f"ERROR: {message}", flush=True)
        if error:
            print(f"  {error}", flush=True)


def load_documents_from_text_folder(text_folder: str, logger: Logger) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Load all text documents from the text folder."""
    doc_texts = {}
    doc_id_mapping = {}
    text_path = Path(text_folder)
    
    if not text_path.exists():
        logger.log_error(f"Text folder not found: {text_folder}")
        raise FileNotFoundError(f"Text folder not found: {text_folder}")
    
    txt_files = sorted(text_path.glob("*.txt"))
    logger.log(f"Found {len(txt_files)} text files")
    
    doc_id_counter = 1
    for txt_file in txt_files:
        doc_name = txt_file.stem
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    doc_texts[doc_id_counter] = text
                    doc_id_mapping[doc_id_counter] = doc_name
                    doc_id_counter += 1
        except Exception as e:
            logger.log_error(f"Error reading {txt_file}", e)
            continue
    
    logger.log(f"Successfully loaded {len(doc_texts)} documents")
    return doc_texts, doc_id_mapping


def create_jsonl_from_documents(doc_texts: Dict[int, str], output_path: str, logger: Logger) -> None:
    """Create a temporary JSONL file for indexing."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id, text in doc_texts.items():
            doc_entry = {
                'docid': doc_id,
                'text': text
            }
            f.write(json.dumps(doc_entry, ensure_ascii=False) + '\n')


def compute_pairwise_similarities(
    doc_texts: Dict[int, str],
    doc_id_mapping: Dict[int, str],
    logger: Logger,
    use_sbert: bool = True,
    use_jaccard: bool = True,
    use_bm25: bool = True,
    chunk_size: int = 200,
    overlap: int = 50,
    index_cache_dir: str = "plagiarism_index_cache"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise similarities for all papers.
    
    Returns:
        Tuple of (sbert_matrix, jaccard_matrix, bm25_matrix) DataFrames
    """
    papers = sorted(doc_id_mapping.keys())
    n_papers = len(papers)
    paper_names = [doc_id_mapping[pid] for pid in papers]
    
    logger.log(f"Computing pairwise similarities for {n_papers} papers...")
    logger.log("This may take a while...")
    
    # Initialize similarity matrices
    sbert_matrix = np.full((n_papers, n_papers), np.nan)
    jaccard_matrix = np.full((n_papers, n_papers), np.nan)
    bm25_matrix = np.full((n_papers, n_papers), np.nan)
    
    # Initialize rankers
    sbert_ranker = None
    jaccard_ranker = None
    bm25_ranker = None
    index = None
    
    # Build/load index and BM25 ranker if needed
    if use_bm25:
        try:
            logger.log("Building/loading index for BM25...")
            tokenizer = RegexTokenizer(r'\w+', lowercase=True)
            stopwords = load_nltk_stopwords()
            
            # Check if index cache exists
            if os.path.exists(index_cache_dir) and os.path.exists(
                os.path.join(index_cache_dir, 'index.json')
            ):
                logger.log("Loading index from cache...")
                index = BasicInvertedIndex()
                index.load(index_cache_dir)
            else:
                logger.log("Creating new index...")
                # Create temporary JSONL file
                temp_jsonl = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8')
                temp_jsonl_path = temp_jsonl.name
                temp_jsonl.close()
                
                create_jsonl_from_documents(doc_texts, temp_jsonl_path, logger)
                
                index = Indexer.create_index(
                    index_type=IndexType.BasicInvertedIndex,
                    dataset_path=temp_jsonl_path,
                    document_preprocessor=tokenizer,
                    stopwords=stopwords,
                    minimum_word_frequency=2,
                    text_key='text',
                    max_docs=-1,
                    id_key='docid'
                )
                
                # Save index
                os.makedirs(index_cache_dir, exist_ok=True)
                index.save(index_cache_dir)
                logger.log("Index saved to cache")
                
                # Clean up temp file
                os.unlink(temp_jsonl_path)
            
            # Create BM25 ranker
            scorer = BM25(index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8})
            bm25_ranker = Ranker(
                index=index,
                document_preprocessor=tokenizer,
                stopwords=stopwords,
                scorer=scorer,
                raw_text_dict=doc_texts
            )
            logger.log("BM25 ranker initialized")
        except Exception as e:
            logger.log_error("Failed to initialize BM25 ranker", e)
            use_bm25 = False
    
    if use_sbert:
        try:
            logger.log("Initializing SBERT ranker...")
            sbert_ranker = SBERTRanker(model_name="all-MiniLM-L6-v2")
            logger.log("SBERT ranker initialized")
        except Exception as e:
            logger.log_error("Failed to initialize SBERT ranker", e)
            use_sbert = False
    
    if use_jaccard:
        try:
            logger.log("Initializing Jaccard ranker...")
            jaccard_ranker = JaccardRanker()
            logger.log("Jaccard ranker initialized")
        except Exception as e:
            logger.log_error("Failed to initialize Jaccard ranker", e)
            use_jaccard = False
    
    # Compute pairwise similarities
    total_pairs = n_papers * (n_papers - 1) // 2
    pair_count = 0
    
    logger.log(f"\nStarting pairwise similarity computation...")
    logger.log(f"Total papers: {n_papers}")
    logger.log(f"Total pairs to compute: {total_pairs}")
    logger.log(f"Using BM25: {use_bm25}, SBERT: {use_sbert}, Jaccard: {use_jaccard}")
    
    for i, paper1_id in enumerate(papers):
        if i % 10 == 0:
            logger.log(f"\nProcessing paper {i+1}/{n_papers}: {doc_id_mapping[paper1_id]}")
        paper1_text = doc_texts[paper1_id]
        
        for j, paper2_id in enumerate(papers):
            if i == j:
                # Self-similarity is 1.0
                if use_bm25:
                    bm25_matrix[i, j] = 1.0
                continue
            
            paper2_text = doc_texts[paper2_id]
            
            # Compute BM25 similarity
            if use_bm25 and bm25_ranker:
                try:
                    # Query paper1 against the index, find paper2's score
                    bm25_results = bm25_ranker.query(paper1_text)
                    # Find paper2_id in results
                    bm25_score = None
                    for matched_doc_id, score in bm25_results:
                        if matched_doc_id == paper2_id:
                            bm25_score = score
                            break
                    
                    if bm25_score is not None:
                        bm25_matrix[i, j] = bm25_score
                except Exception as e:
                    logger.log_error(f"Error computing BM25 similarity for {doc_id_mapping[paper1_id]} vs {doc_id_mapping[paper2_id]}", e)
            
            pair_count += 1
            if pair_count % 50 == 0:  # Log every 50 pairs instead of 10
                progress_pct = (pair_count / total_pairs) * 100
                logger.log(f"Progress: {pair_count}/{total_pairs} pairs ({progress_pct:.1f}%) - Last processed: {doc_id_mapping[paper1_id]} vs {doc_id_mapping[paper2_id]}")
    
    # Normalize BM25 scores using min-max normalization
    if use_bm25:
        logger.log("Normalizing BM25 scores...")
        # Get non-diagonal values for normalization
        mask = ~np.eye(n_papers, dtype=bool)
        bm25_values = bm25_matrix[mask]
        bm25_values = bm25_values[~np.isnan(bm25_values)]
        
        if len(bm25_values) > 0:
            bm25_min = np.nanmin(bm25_values)
            bm25_max = np.nanmax(bm25_values)
            bm25_range = bm25_max - bm25_min
            
            if bm25_range > 0:
                # Normalize: (score - min) / (max - min)
                bm25_matrix_normalized = (bm25_matrix - bm25_min) / bm25_range
                # Preserve diagonal as 1.0 and NaN values
                bm25_matrix_normalized[np.eye(n_papers, dtype=bool)] = 1.0
                bm25_matrix_normalized[np.isnan(bm25_matrix)] = np.nan
                bm25_matrix = bm25_matrix_normalized
                logger.log(f"BM25 scores normalized: min={bm25_min:.2f}, max={bm25_max:.2f}")
            else:
                logger.log("Warning: BM25 range is 0, skipping normalization")
    
    # Create DataFrames
    sbert_df = pd.DataFrame(sbert_matrix, index=paper_names, columns=paper_names)
    jaccard_df = pd.DataFrame(jaccard_matrix, index=paper_names, columns=paper_names)
    bm25_df = pd.DataFrame(bm25_matrix, index=paper_names, columns=paper_names)
    
    logger.log(f"Completed pairwise similarity computation!")
    return sbert_df, jaccard_df, bm25_df


def plot_similarity_heatmap(similarity_matrix: pd.DataFrame, output_file: str, 
                            score_type: str = "SBERT Score"):
    """Create a heatmap of similarity scores."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Choose colormap based on score type
    if 'SBERT' in score_type:
        cmap = 'YlOrRd'
    elif 'BM25' in score_type:
        cmap = 'YlOrRd'  # Use same as SBERT since both are normalized 0-1
    else:
        cmap = 'RdYlBu_r'
    
    sns.heatmap(similarity_matrix, annot=False, fmt='.2f', cmap=cmap, 
                cbar_kws={'label': score_type}, ax=ax, 
                square=True, linewidths=0.5, linecolor='gray')
    
    ax.set_title(f'Pairwise Similarity Matrix - {score_type} (All Papers)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Paper', fontsize=12)
    ax.set_ylabel('Paper', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}", flush=True)
    plt.close()


def save_similarity_matrices(sbert_df: pd.DataFrame, jaccard_df: pd.DataFrame, 
                            bm25_df: pd.DataFrame, output_dir: Path):
    """Save similarity matrices to CSV files."""
    sbert_df.to_csv(output_dir / "sbert_similarity_matrix.csv")
    jaccard_df.to_csv(output_dir / "jaccard_similarity_matrix.csv")
    bm25_df.to_csv(output_dir / "bm25_similarity_matrix.csv")
    print(f"Similarity matrices saved to CSV files in {output_dir}")


def main():
    """Main function."""
    logger = Logger()
    
    text_folder = "corpus/text"
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load all documents
    logger.log("Loading documents from corpus...")
    try:
        doc_texts, doc_id_mapping = load_documents_from_text_folder(text_folder, logger)
    except Exception as e:
        logger.log_error("Failed to load documents", e)
        return
    
    if len(doc_texts) == 0:
        logger.log("No documents found!")
        return
    
    logger.log(f"Loaded {len(doc_texts)} papers")
    
    # Compute pairwise similarities
    sbert_df, jaccard_df, bm25_df = compute_pairwise_similarities(
        doc_texts, doc_id_mapping, logger,
        use_sbert=False,  # Disabled - computationally intensive
        use_jaccard=False,  # Disabled - computationally intensive
        use_bm25=True
    )
    
    # Generate heatmaps
    logger.log("\nGenerating heatmaps...")
    plot_similarity_heatmap(bm25_df, 
                           output_file=str(output_dir / "pairwise_similarity_heatmap_bm25.png"),
                           score_type="BM25 Score (Normalized)")
    
    # Save matrices
    logger.log("\nSaving similarity matrices...")
    bm25_df.to_csv(output_dir / "bm25_similarity_matrix.csv")
    print(f"BM25 similarity matrix saved to {output_dir / 'bm25_similarity_matrix.csv'}", flush=True)
    
    logger.log("\nDone! All pairwise similarity comparisons completed.")


if __name__ == "__main__":
    main()

