"""
Jaccard similarity ranking for plagiarism detection.
Uses Jaccard similarity (set intersection over union) for lexical matching between text chunks.
"""

from typing import List, Tuple, Dict, Optional
import re


class JaccardRanker:
    """
    Lexical similarity ranker using Jaccard similarity (intersection over union).
    """
    
    def __init__(self, case_sensitive: bool = False, normalize: bool = True):
        """
        Initialize Jaccard ranker.
        
        Args:
            case_sensitive: If True, treat words as case-sensitive
            normalize: If True, normalize text before tokenization
        """
        self.case_sensitive = case_sensitive
        self.normalize = normalize
    
    def tokenize(self, text: str) -> set:
        """
        Tokenize text into a set of words.
        
        Args:
            text: Input text
            
        Returns:
            Set of tokens (words)
        """
        if self.normalize:
            # Remove punctuation and split into words
            text = re.sub(r'[^\w\s]', ' ', text)
        
        if not self.case_sensitive:
            text = text.lower()
        
        words = text.split()
        return set(word for word in words if word.strip())
    
    def compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score (0 to 1)
        """
        set1 = self.tokenize(text1)
        set2 = self.tokenize(text2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def rank_chunks_jaccard(
        self,
        query_chunk: str,
        candidate_chunks: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate chunks by Jaccard similarity to query chunk.
        
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
        
        query_tokens = self.tokenize(query_chunk)
        
        similarities = []
        for idx, candidate_chunk in enumerate(candidate_chunks):
            # Skip very short chunks
            if len(candidate_chunk.split()) < 10:
                continue
            
            similarity = self.compute_jaccard_similarity(query_chunk, candidate_chunk)
            if similarity >= similarity_threshold:
                similarities.append((idx, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def find_best_matching_chunk_jaccard(
        self,
        source_chunk: str,
        target_chunks: List[str],
        similarity_threshold: float = 0.3
    ) -> Tuple[int, str, float]:
        """
        Find the best matching chunk using Jaccard similarity.
        
        Args:
            source_chunk: Source chunk text
            target_chunks: List of target chunk texts
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (best_chunk_idx, best_chunk_text, similarity_score)
        """
        if not target_chunks:
            return (0, "", 0.0)
        
        # Rank chunks by Jaccard similarity
        ranked = self.rank_chunks_jaccard(
            source_chunk,
            target_chunks,
            top_k=1,
            similarity_threshold=similarity_threshold
        )
        
        if ranked:
            best_idx, best_score = ranked[0]
            return (best_idx, target_chunks[best_idx], best_score)
        else:
            # Return first chunk with score 0 if no matches found
            return (0, target_chunks[0] if target_chunks else "", 0.0)
    
    def rank_sentences_jaccard(
        self,
        query_sentence: str,
        candidate_sentences: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[int, float]]:
        """
        Rank candidate sentences by Jaccard similarity to query sentence.
        
        Args:
            query_sentence: The query sentence
            candidate_sentences: List of candidate sentences
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples: (sentence_index, similarity_score), sorted by score descending
        """
        return self.rank_chunks_jaccard(
            query_sentence,
            candidate_sentences,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )


def compute_jaccard_similarity_simple(text1: str, text2: str, case_sensitive: bool = False) -> float:
    """
    Simple Jaccard similarity computation (convenience function).
    
    Args:
        text1: First text
        text2: Second text
        case_sensitive: If True, treat words as case-sensitive
        
    Returns:
        Jaccard similarity score (0 to 1)
    """
    ranker = JaccardRanker(case_sensitive=case_sensitive)
    return ranker.compute_jaccard_similarity(text1, text2)

