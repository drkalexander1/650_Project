"""
Analyze plagiarism statistics from PAN corpus evaluation results.
Calculates percentage of plagiarized chunks per source document and creates visualizations.
"""
import os
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from eval_plagiarism import (
    load_documents_from_text_folder,
    extract_chunks,
    Logger
)


def load_plagiarism_results_json(json_file: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load plagiarism results from JSON file.
    
    Args:
        json_file: Path to the JSON file with full plagiarism results
        
    Returns:
        Dictionary mapping source_doc_name to list of (suspicious_doc_name, source_chunk_idx) tuples
    """
    matches_by_source = defaultdict(list)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        json_results = json.load(f)
    
    # Convert from suspicious-doc-centric to source-doc-centric
    for suspicious_doc, matches in json_results.items():
        for match in matches:
            source_doc = match['source_doc']
            source_chunk_idx = match['source_chunk_idx']
            matches_by_source[source_doc].append((suspicious_doc, source_chunk_idx))
    
    return dict(matches_by_source)


def parse_plagiarism_report(report_file: str) -> Dict[str, List[Tuple[str, int]]]:
    """
    Parse the plagiarism report to extract matches (fallback if JSON not available).
    Note: This only extracts examples shown in report, not all matches.
    
    Args:
        report_file: Path to the plagiarism report file
        
    Returns:
        Dictionary mapping source_doc_name to list of (suspicious_doc_name, source_chunk_idx) tuples
    """
    matches_by_source = defaultdict(list)
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all sections with source document matches
    # Pattern: "Source Document: source-documentXXXXX, Chunk #Y"
    pattern = r'Source Document: (source-document\d+), Chunk #(\d+)'
    
    current_suspicious_doc = None
    
    # Split by suspicious document sections
    suspicious_sections = re.split(r'Suspicious Document: (suspicious-document\d+)', content)
    
    for i in range(1, len(suspicious_sections), 2):
        if i + 1 < len(suspicious_sections):
            suspicious_doc = suspicious_sections[i]
            section_content = suspicious_sections[i + 1]
            
            # Find all source document matches in this section
            matches = re.findall(pattern, section_content)
            for source_doc, chunk_idx_str in matches:
                chunk_idx = int(chunk_idx_str)
                matches_by_source[source_doc].append((suspicious_doc, chunk_idx))
    
    return dict(matches_by_source)


def calculate_source_chunk_percentages(
    source_doc_texts: Dict[int, str],
    source_doc_id_mapping: Dict[int, str],
    matches_by_source: Dict[str, List[Tuple[str, int]]],
    logger: Logger
) -> Dict[str, Dict]:
    """
    Calculate percentage of plagiarized chunks for each source document.
    
    Args:
        source_doc_texts: Dictionary mapping doc_id to text
        source_doc_id_mapping: Mapping from doc_id to doc_name
        matches_by_source: Dictionary mapping source_doc_name to list of matches
        logger: Logger instance
        
    Returns:
        Dictionary mapping source_doc_name to stats dict with:
        - total_chunks: total number of chunks
        - plagiarized_chunks: number of unique chunks that were plagiarized
        - percentage: percentage of plagiarized chunks
    """
    logger.log("Calculating chunk percentages for source documents...")
    
    source_stats = {}
    
    # Create reverse mapping: doc_name -> doc_id
    name_to_id = {name: doc_id for doc_id, name in source_doc_id_mapping.items()}
    
    for source_doc_name, matches in matches_by_source.items():
        # Get the source document text
        source_doc_id = name_to_id.get(source_doc_name)
        if source_doc_id is None:
            logger.log(f"Warning: Source document {source_doc_name} not found in loaded documents")
            continue
        
        source_text = source_doc_texts.get(source_doc_id, "")
        if not source_text:
            continue
        
        # Extract all chunks from source document
        chunks = extract_chunks(source_text, chunk_size=200, overlap=50)
        total_chunks = len([c for c in chunks if len(c.split()) >= 20])  # Only count valid chunks
        
        # Count unique plagiarized chunks (chunk indices that appear in matches)
        plagiarized_chunk_indices = set()
        for suspicious_doc, chunk_idx in matches:
            if chunk_idx < total_chunks:  # Make sure chunk index is valid
                plagiarized_chunk_indices.add(chunk_idx)
        
        plagiarized_chunks = len(plagiarized_chunk_indices)
        percentage = (plagiarized_chunks / total_chunks * 100) if total_chunks > 0 else 0.0
        
        source_stats[source_doc_name] = {
            'total_chunks': total_chunks,
            'plagiarized_chunks': plagiarized_chunks,
            'percentage': percentage,
            'num_matches': len(matches)
        }
    
    # Also process source documents that had no matches (0% plagiarized)
    for doc_id, doc_name in source_doc_id_mapping.items():
        if doc_name not in source_stats:
            source_text = source_doc_texts.get(doc_id, "")
            if source_text:
                chunks = extract_chunks(source_text, chunk_size=200, overlap=50)
                total_chunks = len([c for c in chunks if len(c.split()) >= 20])
                source_stats[doc_name] = {
                    'total_chunks': total_chunks,
                    'plagiarized_chunks': 0,
                    'percentage': 0.0,
                    'num_matches': 0
                }
    
    logger.log(f"Calculated statistics for {len(source_stats)} source documents")
    return source_stats


def calculate_percentage_distribution(source_stats: Dict[str, Dict]) -> Dict[str, int]:
    """
    Calculate distribution of plagiarism percentages across source documents.
    
    Args:
        source_stats: Dictionary mapping source_doc_name to stats
        
    Returns:
        Dictionary mapping percentage ranges to count of documents
    """
    # Define percentage bins: 0-10%, 10-20%, ..., 90-100%
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    distribution = defaultdict(int)
    
    for doc_name, stats in source_stats.items():
        percentage = stats['percentage']
        # Find which bin this percentage falls into
        for i in range(len(bins) - 1):
            if bins[i] <= percentage < bins[i + 1]:
                bin_label = f"{bins[i]}-{bins[i+1]}%"
                distribution[bin_label] += 1
                break
        # Handle exactly 100%
        if percentage == 100:
            distribution["90-100%"] += 1
    
    return dict(distribution)


def save_statistics(source_stats: Dict[str, Dict], distribution: Dict[str, int], 
                   output_file: str, logger: Logger):
    """
    Save statistics to a text file.
    
    Args:
        source_stats: Dictionary mapping source_doc_name to stats
        distribution: Dictionary mapping percentage ranges to counts
        output_file: Path to output file
        logger: Logger instance
    """
    logger.log(f"Saving statistics to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PLAGIARISM STATISTICS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total source documents analyzed: {len(source_stats)}\n")
        
        # Calculate summary statistics
        percentages = [stats['percentage'] for stats in source_stats.values()]
        if percentages:
            f.write(f"\nSummary Statistics:\n")
            f.write(f"  Mean percentage of plagiarized chunks: {np.mean(percentages):.2f}%\n")
            f.write(f"  Median percentage: {np.median(percentages):.2f}%\n")
            f.write(f"  Standard deviation: {np.std(percentages):.2f}%\n")
            f.write(f"  Min percentage: {min(percentages):.2f}%\n")
            f.write(f"  Max percentage: {max(percentages):.2f}%\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write("PERCENTAGE DISTRIBUTION\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort distribution by percentage range
        sorted_bins = sorted(distribution.items(), key=lambda x: float(x[0].split('-')[0]))
        for bin_range, count in sorted_bins:
            percentage_of_docs = (count / len(source_stats) * 100) if source_stats else 0
            f.write(f"  {bin_range:12} : {count:4} documents ({percentage_of_docs:5.2f}% of all source docs)\n")
        
        f.write(f"\n" + "=" * 80 + "\n")
        f.write("DETAILED STATISTICS BY SOURCE DOCUMENT\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by percentage (descending)
        sorted_stats = sorted(source_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        
        f.write(f"{'Source Document':<40} {'Total Chunks':<15} {'Plagiarized':<15} {'Percentage':<15} {'Matches':<10}\n")
        f.write("-" * 95 + "\n")
        
        for doc_name, stats in sorted_stats:
            f.write(f"{doc_name:<40} {stats['total_chunks']:<15} {stats['plagiarized_chunks']:<15} "
                   f"{stats['percentage']:<15.2f} {stats['num_matches']:<10}\n")
    
    logger.log(f"Statistics saved to {output_file}")


def create_visualizations(source_stats: Dict[str, Dict], distribution: Dict[str, int],
                         output_dir: Path, logger: Logger):
    """
    Create visualizations of plagiarism statistics.
    
    Args:
        source_stats: Dictionary mapping source_doc_name to stats
        distribution: Dictionary mapping percentage ranges to counts
        output_dir: Directory to save visualizations
        logger: Logger instance
    """
    logger.log("Creating visualizations...")
    
    # 1. Histogram of percentage distribution
    percentages = [stats['percentage'] for stats in source_stats.values()]
    
    plt.figure(figsize=(12, 6))
    plt.hist(percentages, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Percentage of Plagiarized Chunks (%)', fontsize=12)
    plt.ylabel('Number of Source Documents', fontsize=12)
    plt.title('Distribution of Plagiarized Chunk Percentages Across Source Documents', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plagiarism_percentage_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("  Saved: plagiarism_percentage_histogram.png")
    
    # 2. Bar chart of percentage bins
    sorted_bins = sorted(distribution.items(), key=lambda x: float(x[0].split('-')[0]))
    bin_ranges = [x[0] for x in sorted_bins]
    bin_counts = [x[1] for x in sorted_bins]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(bin_ranges, bin_counts, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Percentage Range', fontsize=12)
    plt.ylabel('Number of Source Documents', fontsize=12)
    plt.title('Distribution of Source Documents by Plagiarism Percentage Range', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plagiarism_percentage_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("  Saved: plagiarism_percentage_distribution.png")
    
    # 3. Box plot of percentages
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
    
    # 4. Cumulative distribution
    sorted_percentages = sorted(percentages)
    cumulative = np.arange(1, len(sorted_percentages) + 1) / len(sorted_percentages) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_percentages, cumulative, linewidth=2)
    plt.xlabel('Percentage of Plagiarized Chunks (%)', fontsize=12)
    plt.ylabel('Cumulative Percentage of Documents (%)', fontsize=12)
    plt.title('Cumulative Distribution of Plagiarized Chunk Percentages', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plagiarism_percentage_cumulative.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.log("  Saved: plagiarism_percentage_cumulative.png")
    
    logger.log("All visualizations created successfully")


def main():
    """
    Main function to analyze plagiarism statistics.
    """
    # Configuration
    source_folder = "pan-plagiarism-corpus-2011/external-detection-corpus/source-document/part3"
    report_file = "pan_plagiarism_report_part3_filtered.txt"
    output_dir = Path("eval_results")
    log_file = "plagiarism_analysis.log"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Initialize logger
    logger = Logger(log_file)
    
    try:
        logger.log("=" * 80)
        logger.log("PLAGIARISM STATISTICS ANALYSIS")
        logger.log("=" * 80)
        
        # Step 1: Load source documents
        logger.log("\nStep 1: Loading source documents...")
        try:
            source_doc_texts, source_doc_id_mapping = load_documents_from_text_folder(
                source_folder, logger, recursive=False, max_lines=2500
            )
        except Exception as e:
            logger.log_error("Failed to load source documents", e)
            raise
        
        logger.log(f"Loaded {len(source_doc_texts)} source documents")
        
        # Step 2: Load plagiarism results (prefer JSON if available, else parse report)
        json_results_file = "pan_plagiarism_results_part3_filtered.json"
        if os.path.exists(json_results_file):
            logger.log("\nStep 2: Loading full plagiarism results from JSON...")
            try:
                matches_by_source = load_plagiarism_results_json(json_results_file)
                logger.log(f"Found matches for {len(matches_by_source)} source documents")
                total_matches = sum(len(matches) for matches in matches_by_source.values())
                logger.log(f"Total matches found: {total_matches}")
            except Exception as e:
                logger.log_error("Failed to load JSON results, falling back to report parsing", e)
                logger.log("\nStep 2: Parsing plagiarism report (limited to examples only)...")
                matches_by_source = parse_plagiarism_report(report_file)
                logger.log(f"WARNING: Report only contains examples, not all matches!")
                logger.log(f"Found matches for {len(matches_by_source)} source documents")
        else:
            logger.log("\nStep 2: JSON results file not found, parsing report (limited to examples only)...")
            logger.log(f"WARNING: Report only contains examples, not all matches!")
            logger.log(f"To get complete statistics, re-run eval_plagiarism.py to generate JSON file.")
            try:
                matches_by_source = parse_plagiarism_report(report_file)
                logger.log(f"Found matches for {len(matches_by_source)} source documents")
            except Exception as e:
                logger.log_error("Failed to parse plagiarism report", e)
                raise
        
        # Step 3: Calculate chunk percentages
        logger.log("\nStep 3: Calculating chunk percentages...")
        try:
            source_stats = calculate_source_chunk_percentages(
                source_doc_texts,
                source_doc_id_mapping,
                matches_by_source,
                logger
            )
        except Exception as e:
            logger.log_error("Failed to calculate percentages", e)
            raise
        
        # Step 4: Calculate distribution
        logger.log("\nStep 4: Calculating percentage distribution...")
        distribution = calculate_percentage_distribution(source_stats)
        logger.log("Distribution calculated:")
        sorted_bins = sorted(distribution.items(), key=lambda x: float(x[0].split('-')[0]))
        for bin_range, count in sorted_bins:
            logger.log(f"  {bin_range}: {count} documents")
        
        # Step 5: Save statistics
        logger.log("\nStep 5: Saving statistics...")
        stats_file = output_dir / "plagiarism_statistics.txt"
        save_statistics(source_stats, distribution, str(stats_file), logger)
        
        # Also save as JSON for easy access
        json_file = output_dir / "plagiarism_statistics.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_stats': source_stats,
                'distribution': distribution,
                'summary': {
                    'total_source_docs': len(source_stats),
                    'mean_percentage': float(np.mean([s['percentage'] for s in source_stats.values()])),
                    'median_percentage': float(np.median([s['percentage'] for s in source_stats.values()])),
                    'std_percentage': float(np.std([s['percentage'] for s in source_stats.values()]))
                }
            }, f, indent=2, ensure_ascii=False)
        logger.log(f"Statistics also saved as JSON to {json_file}")
        
        # Step 6: Create visualizations
        logger.log("\nStep 6: Creating visualizations...")
        try:
            create_visualizations(source_stats, distribution, output_dir, logger)
        except Exception as e:
            logger.log_error("Failed to create visualizations", e)
            raise
        
        logger.log("\n" + "=" * 80)
        logger.log("ANALYSIS COMPLETE")
        logger.log("=" * 80)
        logger.log(f"\nAll results saved to: {output_dir}/")
        logger.log(f"  - Statistics: {stats_file}")
        logger.log(f"  - JSON data: {json_file}")
        logger.log(f"  - Visualizations: *.png files")
        
    except Exception as e:
        logger.log_error("Fatal error in analysis", e)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
