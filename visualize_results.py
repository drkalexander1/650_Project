#!/usr/bin/env python3
"""
Visualization script for plagiarism detection query results.
Generates graphs and tables from query_results.json files.
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

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_all_query_results(base_dir: str = ".") -> Dict[str, Dict]:
    """
    Load all query_results.json files from query_results_* directories.
    
    Returns:
        Dictionary mapping query_name to query results
    """
    results = {}
    base_path = Path(base_dir)
    
    for query_dir in base_path.glob("query_results_*"):
        json_file = query_dir / "query_results.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    query_name = data.get('query_name', query_dir.name)
                    results[query_name] = data
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return results


def create_similarity_matrix(results: Dict[str, Dict], score_type: str = 'sbert_score') -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a similarity matrix from query results.
    
    Args:
        results: Dictionary of query results
        score_type: Which score to use ('sbert_score', 'score', 'jaccard_score', 'bm25_score')
    
    Returns:
        Tuple of (similarity matrix DataFrame, list of paper names)
    """
    # Collect all unique papers
    all_papers = set()
    for query_data in results.values():
        all_papers.add(query_data['query_name'])
        for result in query_data.get('results', []):
            all_papers.add(result['paper_name'])
    
    all_papers = sorted(list(all_papers))
    n_papers = len(all_papers)
    
    # Initialize matrix with NaN
    matrix = np.full((n_papers, n_papers), np.nan)
    paper_to_idx = {paper: idx for idx, paper in enumerate(all_papers)}
    
    # Fill matrix from query results
    for query_data in results.values():
        query_name = query_data['query_name']
        query_idx = paper_to_idx.get(query_name)
        
        if query_idx is not None:
            for result in query_data.get('results', []):
                matched_name = result['paper_name']
                matched_idx = paper_to_idx.get(matched_name)
                
                if matched_idx is not None:
                    score = result.get(score_type, np.nan)
                    if not np.isnan(score):
                        # Store the maximum score if multiple matches exist
                        if np.isnan(matrix[query_idx, matched_idx]) or matrix[query_idx, matched_idx] < score:
                            matrix[query_idx, matched_idx] = score
    
    # Create DataFrame
    df = pd.DataFrame(matrix, index=all_papers, columns=all_papers)
    
    return df, all_papers


def plot_similarity_heatmap(similarity_matrix: pd.DataFrame, output_file: str = "similarity_heatmap.png", 
                            score_type: str = "SBERT Score"):
    """
    Create a heatmap of similarity scores.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use a colormap that works well for similarity scores
    cmap = 'YlOrRd' if score_type == 'SBERT Score' else 'RdYlBu_r'
    
    # Create heatmap
    sns.heatmap(similarity_matrix, annot=False, fmt='.2f', cmap=cmap, 
                cbar_kws={'label': score_type}, ax=ax, 
                square=True, linewidths=0.5, linecolor='gray')
    
    ax.set_title(f'Similarity Matrix - {score_type}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Matched Paper', fontsize=12)
    ax.set_ylabel('Query Paper', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    plt.close()


def plot_top_matches_bar_chart(results: Dict[str, Dict], output_file: str = "top_matches_bar.png",
                               score_type: str = 'sbert_score', top_k: int = 3):
    """
    Create a bar chart showing top matches for each queried paper.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data
    query_papers = []
    match_papers = []
    scores = []
    positions = []
    
    x_pos = 0
    for query_name in sorted(results.keys()):
        query_data = results[query_name]
        top_results = sorted(query_data.get('results', []), 
                           key=lambda x: x.get(score_type, 0), 
                           reverse=True)[:top_k]
        
        for rank, result in enumerate(top_results):
            query_papers.append(query_name)
            match_papers.append(result['paper_name'])
            scores.append(result.get(score_type, 0))
            positions.append(x_pos + rank * 0.25)
        
        x_pos += top_k + 1
    
    # Create grouped bar chart
    colors = plt.cm.Set3(np.linspace(0, 1, top_k))
    bars = ax.bar(positions, scores, width=0.2, color=colors[:len(scores)])
    
    # Set x-axis labels
    ax.set_xticks([pos + (top_k-1)*0.25/2 for pos in range(0, len(results)*(top_k+1), top_k+1)])
    ax.set_xticklabels(sorted(results.keys()), rotation=45, ha='right')
    
    ax.set_ylabel(f'{score_type.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Top {top_k} Matches per Query Paper', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax.legend([f'Rank {i+1}' for i in range(top_k)], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_file}")
    plt.close()


def create_summary_table(results: Dict[str, Dict], output_file: str = "summary_table.csv"):
    """
    Create a summary table with statistics for each query.
    """
    rows = []
    
    for query_name, query_data in sorted(results.items()):
        query_results = query_data.get('results', [])
        
        if not query_results:
            continue
        
        sbert_scores = [r.get('sbert_score', 0) for r in query_results]
        jaccard_scores = [r.get('jaccard_score', 0) for r in query_results]
        bm25_scores = [r.get('bm25_score', 0) for r in query_results]
        overall_scores = [r.get('score', 0) for r in query_results]
        
        rows.append({
            'Query Paper': query_name,
            'Num Matches': len(query_results),
            'Max SBERT': max(sbert_scores) if sbert_scores else 0,
            'Mean SBERT': np.mean(sbert_scores) if sbert_scores else 0,
            'Max Jaccard': max(jaccard_scores) if jaccard_scores else 0,
            'Mean Jaccard': np.mean(jaccard_scores) if jaccard_scores else 0,
            'Max Overall': max(overall_scores) if overall_scores else 0,
            'Mean Overall': np.mean(overall_scores) if overall_scores else 0,
            'Top Match': query_results[0]['paper_name'] if query_results else 'N/A',
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Summary table saved to {output_file}")
    return df


def create_detailed_results_table(results: Dict[str, Dict], output_file: str = "detailed_results.csv"):
    """
    Create a detailed table with all query-match pairs.
    """
    rows = []
    
    for query_name, query_data in sorted(results.items()):
        for rank, result in enumerate(query_data.get('results', []), 1):
            rows.append({
                'Query Paper': query_name,
                'Rank': rank,
                'Matched Paper': result['paper_name'],
                'Overall Score': result.get('score', 0),
                'SBERT Score': result.get('sbert_score', 0),
                'Jaccard Score': result.get('jaccard_score', 0),
                'BM25 Score': result.get('bm25_score', 0),
                'Num Chunks': len(result.get('chunks', [])),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Detailed results table saved to {output_file}")
    return df


def plot_score_distribution(results: Dict[str, Dict], output_file: str = "score_distribution.png"):
    """
    Create histograms showing the distribution of different score types.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Score Distributions', fontsize=16, fontweight='bold')
    
    # Collect all scores
    sbert_scores = []
    jaccard_scores = []
    overall_scores = []
    bm25_scores = []
    
    for query_data in results.values():
        for result in query_data.get('results', []):
            sbert_scores.append(result.get('sbert_score', 0))
            jaccard_scores.append(result.get('jaccard_score', 0))
            overall_scores.append(result.get('score', 0))
            bm25_score = result.get('bm25_score', 0)
            # Only include BM25 scores that are reasonable (not extremely negative)
            if bm25_score > -10000:
                bm25_scores.append(bm25_score)
    
    # Plot distributions
    axes[0, 0].hist(sbert_scores, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('SBERT Score Distribution')
    axes[0, 0].set_xlabel('SBERT Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(jaccard_scores, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title('Jaccard Score Distribution')
    axes[0, 1].set_xlabel('Jaccard Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].hist(overall_scores, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('Overall Score Distribution')
    axes[1, 0].set_xlabel('Overall Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    
    if bm25_scores:
        axes[1, 1].hist(bm25_scores, bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[1, 1].set_title('BM25 Score Distribution (filtered)')
        axes[1, 1].set_xlabel('BM25 Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No BM25 scores\nin range', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('BM25 Score Distribution')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Score distribution plot saved to {output_file}")
    plt.close()


def main():
    """Main function to generate all visualizations and tables."""
    print("Loading query results...")
    results = load_all_query_results()
    
    if not results:
        print("No query results found! Make sure query_results_* directories exist.")
        return
    
    print(f"Loaded {len(results)} query result sets")
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate similarity matrix and heatmap for SBERT
    print("\nGenerating SBERT similarity matrix...")
    similarity_matrix_sbert, paper_names = create_similarity_matrix(results, score_type='sbert_score')
    plot_similarity_heatmap(similarity_matrix_sbert, 
                           output_file=str(output_dir / "similarity_heatmap_sbert.png"),
                           score_type="SBERT Score")
    
    # Generate similarity matrix and heatmap for Jaccard
    print("\nGenerating Jaccard similarity matrix...")
    similarity_matrix_jaccard, _ = create_similarity_matrix(results, score_type='jaccard_score')
    plot_similarity_heatmap(similarity_matrix_jaccard, 
                           output_file=str(output_dir / "similarity_heatmap_jaccard.png"),
                           score_type="Jaccard Score")
    
    # Generate bar chart
    print("\nGenerating bar chart...")
    plot_top_matches_bar_chart(results, 
                               output_file=str(output_dir / "top_matches_bar.png"),
                               score_type='sbert_score', 
                               top_k=3)
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_df = create_summary_table(results, output_file=str(output_dir / "summary_table.csv"))
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Generate detailed results table
    print("\nGenerating detailed results table...")
    detailed_df = create_detailed_results_table(results, output_file=str(output_dir / "detailed_results.csv"))
    
    # Generate score distribution plots
    print("\nGenerating score distribution plots...")
    plot_score_distribution(results, output_file=str(output_dir / "score_distribution.png"))
    
    print(f"\nAll visualizations and tables saved to '{output_dir}' directory!")


if __name__ == "__main__":
    main()

