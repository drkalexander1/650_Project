# Plagiarism Detection System - Quick Start Guide

## Master File

**`codetorun.py`** is the main file to run the complete plagiarism detection pipeline.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

```bash
python codetorun.py
```

That's it! The system will:
- Load all `.txt` files from `corpus/text/`
- Build an index (cached for future runs)
- Detect plagiarism using BM25, SBERT, and Jaccard
- Generate `plagiarism_report.txt` with results

## Prerequisites

1. **Corpus**: Text files in `corpus/text/` directory (each `.txt` file = one paper)
   - Already have files? You're good to go!
   - Need to download? Use `download_corpus.py` or `download_corpus_arxiv.py`

2. **Stopwords file**: `stopwords.txt` (already created for you)

## Configuration

Edit settings in `codetorun.py` `main()` function:

```python
# In main() function around line 662:
similarity_threshold = 0.3  # Lower = more matches
top_k = 5                   # Number of matches per chunk

# Ranking methods (around line 670):
use_sbert = True            # Enable SBERT semantic similarity
use_jaccard = True          # Enable Jaccard lexical similarity
hybrid_mode = True          # Combine methods
bm25_weight = 0.3           # Weight for BM25
sbert_weight = 0.4          # Weight for SBERT  
jaccard_weight = 0.3         # Weight for Jaccard
```

## Output Files

- **`plagiarism_report.txt`** - Detailed findings with chunk pairs
- **`plagiarism_detection.log`** - Execution log
- **`plagiarism_index_cache/`** - Cached index (speeds up future runs)

## Example Workflow

```bash
# Step 1: Download corpus (if you don't have one)
python download_corpus_arxiv.py --author "Author Name" --limit 20

# Step 2: Run plagiarism detection
python codetorun.py

# Step 3: View results
cat plagiarism_report.txt
```

## Troubleshooting

**Problem**: "Stopwords file not found"
- **Solution**: `stopwords.txt` should exist (already created for you)

**Problem**: "No documents found"
- **Solution**: Check that `corpus/text/` contains `.txt` files

**Problem**: SBERT model download takes time
- **Solution**: First run downloads ~80MB model. Subsequent runs are faster.

**Problem**: Memory errors
- **Solution**: Reduce `top_k` value or process fewer documents

## What Each Method Does

- **BM25**: Fast lexical matching (word-based)
- **SBERT**: Semantic similarity (catches paraphrases)
- **Jaccard**: Word overlap similarity (catches exact/near-exact matches)

Using all three together gives the best results!

