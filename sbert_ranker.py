"""
SBERT-based semantic similarity ranking for plagiarism detection.
Uses Sentence-BERT to compute semantic similarity between text chunks.
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


class SBERTRanker:
    """
    Semantic similarity ranker using Sentence-BERT embeddings.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize SBERT ranker.
        
        Args:
            model_name: Name of the SBERT model to use
                - 'all-MiniLM-L6-v2': Fast, good quality (default)
                - 'all-mpnet-base-v2': Higher quality, slower
                - 'paraphrase-MiniLM-L6-v2': Optimized for paraphrase detection
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        if not SBERT_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Loading SBERT model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded successfully")
        
        # Cache for document embeddings
        self._doc_embeddings_cache: Dict[int, np.ndarray] = {}
        self._chunk_embeddings_cache: Dict[Tuple[int, int], np.ndarray] = {}
    
    def encode_sentences(self, sentences: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: List of sentence strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings (shape: [num_sentences, embedding_dim])
        """
        if not sentences:
            return np.array([])
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def rank_chunks_semantically(
        self,
        query_chunk: str,
        candidate_chunks: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate chunks by semantic similarity to query chunk.
        
        Args:
            query_chunk: The query chunk text
            candidate_chunks: List of candidate chunk texts
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples: (chunk_index, similarity_score), sorted by score descending
        """
        if not candidate_chunks:
            return []
        
        # Encode query chunk
        query_embedding = self.encode_sentences([query_chunk])[0]
        
        # Encode all candidate chunks
        candidate_embeddings = self.encode_sentences(candidate_chunks)
        
        # Compute similarities
        similarities = []
        for idx, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate_emb)
            if similarity >= similarity_threshold:
                similarities.append((idx, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_best_matching_chunk_semantic(
        self,
        source_chunk: str,
        target_chunks: List[str],
        similarity_threshold: float = 0.3
    ) -> Tuple[int, str, float]:
        """
        Find the best matching chunk using semantic similarity.
        
        Args:
            source_chunk: Source chunk text
            target_chunks: List of target chunk texts
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (best_chunk_idx, best_chunk_text, similarity_score)
        """
        if not target_chunks:
            return (0, "", 0.0)
        
        # Rank chunks semantically
        ranked = self.rank_chunks_semantically(
            source_chunk,
            target_chunks,
            top_k=1,
            similarity_threshold=similarity_threshold
        )
        
        if ranked:
            best_idx, best_score = ranked[0]
            return (best_idx, target_chunks[best_idx], best_score)
        else:
            return (0, target_chunks[0] if target_chunks else "", 0.0)
    
    def rank_sentences_semantically(
        self,
        query_sentence: str,
        candidate_sentences: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate sentences by semantic similarity to query sentence.
        
        Args:
            query_sentence: The query sentence
            candidate_sentences: List of candidate sentences
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples: (sentence_index, similarity_score), sorted by score descending
        """
        return self.rank_chunks_semantically(
            query_sentence,
            candidate_sentences,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]\s+', text)
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences


def hybrid_rank(
    bm25_scores: List[Tuple[int, float]],
    sbert_scores: List[Tuple[int, float]],
    bm25_weight: float = 0.5,
    sbert_weight: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Combine BM25 and SBERT scores using weighted combination.
    
    Args:
        bm25_scores: List of (doc_id, bm25_score) tuples
        sbert_scores: List of (doc_id, sbert_score) tuples
        bm25_weight: Weight for BM25 scores (default: 0.5)
        sbert_weight: Weight for SBERT scores (default: 0.5)
        
    Returns:
        Combined ranked list of (doc_id, combined_score) tuples
    """
    # Normalize weights
    total_weight = bm25_weight + sbert_weight
    bm25_weight /= total_weight
    sbert_weight /= total_weight
    
    # Convert to dictionaries for easier lookup
    bm25_dict = {doc_id: score for doc_id, score in bm25_scores}
    sbert_dict = {doc_id: score for doc_id, score in sbert_scores}
    
    # Get all unique document IDs
    all_doc_ids = set(bm25_dict.keys()) | set(sbert_dict.keys())
    
    # Normalize scores to [0, 1] range using min-max normalization
    # This handles negative BM25 scores correctly
    bm25_values = list(bm25_dict.values()) if bm25_dict else []
    if bm25_values:
        bm25_min = min(bm25_values)
        bm25_max = max(bm25_values)
        bm25_range = bm25_max - bm25_min
    else:
        bm25_min = 0.0
        bm25_max = 1.0
        bm25_range = 1.0
    
    sbert_values = list(sbert_dict.values()) if sbert_dict else []
    if sbert_values:
        sbert_min = min(sbert_values)
        sbert_max = max(sbert_values)
        sbert_range = sbert_max - sbert_min
    else:
        sbert_min = 0.0
        sbert_max = 1.0
        sbert_range = 1.0
    
    # Combine scores
    combined_scores = []
    for doc_id in all_doc_ids:
        # Min-max normalization: (score - min) / (max - min)
        bm25_raw = bm25_dict.get(doc_id, bm25_min)
        bm25_score = (bm25_raw - bm25_min) / bm25_range if bm25_range > 0 else 0.0
        
        sbert_raw = sbert_dict.get(doc_id, sbert_min)
        sbert_score = (sbert_raw - sbert_min) / sbert_range if sbert_range > 0 else 0.0
        
        combined_score = (bm25_weight * bm25_score) + (sbert_weight * sbert_score)
        combined_scores.append((doc_id, combined_score))
    
    # Sort by combined score (descending)
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    return combined_scores

