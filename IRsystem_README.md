# Plagiarism Detection System

A comprehensive plagiarism detection system for academic papers using information retrieval techniques. This system can detect copy-pasted and rephrased sections between papers that are not properly cited.

## Overview

This project implements a multi-stage plagiarism detection pipeline:
1. **Corpus Download**: Download papers from arXiv for specific authors
2. **Preprocessing**: Tokenize and preprocess documents
3. **Indexing**: Build inverted indices for efficient search
4. **Ranking**: Use multiple ranking algorithms (BM25, SBERT, Jaccard) to find similar content
5. **Detection**: Detect plagiarism by comparing document chunks
6. **Evaluation**: Analyze results and generate statistics

## Files Description

### Core Detection Scripts

#### `codetorun.py`
Main plagiarism detection system for academic papers. Supports multiple operation modes:
- **Detection Mode**: Detects plagiarism across all documents in a corpus
- **Query Mode**: Query a single paper against the corpus
- **Evaluation Mode**: Generate evaluation queries for a set of papers
- **Auto-Query Mode**: Automatically query multiple papers from the corpus

**Features:**
- Multiple ranking methods: BM25, SBERT (semantic similarity), Jaccard (lexical similarity)
- Hybrid ranking combining multiple methods
- Chunk-based detection with configurable overlap
- Generates detailed reports with YES/NO answers and examples
- Creates visualizations and statistics

**Usage:**
```bash
# Detection mode (default)
python codetorun.py --mode detection

# Query a single paper
python codetorun.py --mode query --query-paper path/to/paper.txt --query-name "Paper_Name"

# Auto-query mode (query N papers automatically)
python codetorun.py --mode auto_query --auto-query-count 5

# Evaluation mode
python codetorun.py --mode evaluation --evaluation-papers-dir path/to/papers/
```

**Configuration:**
- Corpus folder: `corpus_arxiv_Edward_Witten` (configurable in code)
- Similarity thresholds: BM25 (0.5), SBERT (0.65), Jaccard (0.4)
- Chunk size: 200 words with 50 word overlap
- Output: Results saved to `results_edward_witten/`

#### `eval_plagiarism.py`
Plagiarism detection system specifically designed for the PAN Plagiarism Corpus 2011. Detects plagiarism between source documents and suspicious documents.

**Features:**
- Processes source documents and suspicious documents separately
- Builds index from source documents only
- Queries suspicious documents against source index
- Filters documents by line count (default: max 2500 lines)
- Generates detailed reports with reuse detection (YES/NO)

**Usage:**
```bash
python eval_plagiarism.py
```

**Configuration:**
- Source folder: `pan-plagiarism-corpus-2011/external-detection-corpus/source-document/part3`
- Suspicious folder: `pan-plagiarism-corpus-2011/intrinsic-detection-corpus/suspicious-document/part3`
- Output: `pan_plagiarism_report_part3_filtered.txt` and `pan_plagiarism_results_part3_filtered.json`

#### `analyze_plagiarism_stats.py`
Analyzes plagiarism statistics from PAN corpus evaluation results. Calculates the percentage of plagiarized chunks per source document and creates visualizations.

**Features:**
- Calculates plagiarism percentages for each source document
- Generates distribution statistics
- Creates multiple visualizations (histograms, box plots, cumulative distributions)
- Saves detailed statistics reports

**Usage:**
```bash
python analyze_plagiarism_stats.py
```

**Output:**
- Statistics report: `eval_results/plagiarism_statistics.txt`
- JSON data: `eval_results/plagiarism_statistics.json`
- Visualizations: `eval_results/*.png`

### Corpus Management

#### `download_corpus_arxiv.py`
Downloads papers from arXiv for a specific author and extracts text from PDFs.

**Features:**
- Searches arXiv for papers by a specific author
- Downloads both v1 (first version) and latest version of each paper
- Extracts text from PDFs using pdfplumber or PyPDF2
- Filters papers by minimum word count (default: 500 words)
- Saves metadata, PDFs, and extracted text
- Handles author name variations

**Usage:**
```bash
# Download papers for an author
python download_corpus_arxiv.py --author "Edward Witten"

# Limit number of papers (for testing)
python download_corpus_arxiv.py --author "Edward Witten" --limit 10

# Custom output directory
python download_corpus_arxiv.py --author "Edward Witten" --output my_corpus

# Adjust minimum word threshold
python download_corpus_arxiv.py --author "Edward Witten" --min-words 1000

# Save abstract-only papers
python download_corpus_arxiv.py --author "Edward Witten" --save-abstracts
```

**Output Structure:**
```
corpus_arxiv_[Author_Name]/
├── pdf_arxiv/          # PDF files
├── text_arxiv/         # Extracted text files
├── metadata_arxiv/    # JSON metadata files
└── corpus_summary_arxiv.json  # Summary of downloaded papers
```

**Dependencies:**
- `feedparser`: For arXiv API access
- `pdfplumber` or `PyPDF2`: For PDF text extraction

### Core Modules

#### `preprocessing.py`
Tokenization and preprocessing module for text documents.

**Classes:**
- `Tokenizer`: Base tokenizer class
- `RegexTokenizer`: Tokenizer using NLTK's RegexpTokenizer with regex patterns
- `load_nltk_stopwords()`: Loads English stopwords from NLTK

**Usage:**
```python
from preprocessing import RegexTokenizer, load_nltk_stopwords

# Initialize tokenizer
tokenizer = RegexTokenizer(r'\w+', lowercase=True)

# Tokenize text
tokens = tokenizer.tokenize("This is a sample text.")

# Load stopwords
stopwords = load_nltk_stopwords()
```

**Dependencies:**
- `nltk`: For tokenization and stopwords

#### `ranker.py`
Ranking module implementing various relevance scoring algorithms.

**Classes:**
- `Ranker`: Main ranker class that uses a RelevanceScorer
- `RelevanceScorer`: Base class for relevance scoring algorithms
- `WordCountCosineSimilarity`: Unnormalized cosine similarity on word counts
- `DirichletLM`: Dirichlet Language Model with smoothing parameter μ
- `BM25`: BM25 ranking algorithm (default: b=0.75, k1=1.2, k3=8)
- `PivotedNormalization`: Pivoted normalization ranking (default: b=0.2)
- `TF_IDF`: Term Frequency-Inverse Document Frequency scoring

**Usage:**
```python
from ranker import Ranker, BM25
from preprocessing import RegexTokenizer

# Initialize components
tokenizer = RegexTokenizer(r'\w+', lowercase=True)
stopwords = load_nltk_stopwords()
scorer = BM25(index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8})

# Create ranker
ranker = Ranker(
    index=index,
    document_preprocessor=tokenizer,
    stopwords=stopwords,
    scorer=scorer,
    raw_text_dict=doc_texts
)

# Query
results = ranker.query("quantum field theory")
```

#### `relevance.py`
Evaluation metrics module for measuring ranking performance.

**Functions:**
- `map_score()`: Calculates Mean Average Precision (MAP) score
- `ndcg_score()`: Calculates Normalized Discounted Cumulative Gain (NDCG) score
- `run_relevance_tests()`: Runs evaluation tests on a ranker using relevance data

**Usage:**
```python
from relevance import run_relevance_tests

# Evaluate ranker performance
results = run_relevance_tests('relevance.csv', ranker)
print(f"MAP: {results['map']:.4f}")
print(f"NDCG: {results['ndcg']:.4f}")
```

**Relevance Data Format:**
CSV file with columns: `query`, `docid`, `rel`
- `rel`: Relevance score (1 = marginally relevant, 2 = very relevant)

## Installation

### Prerequisites
```bash
pip install nltk
pip install feedparser
pip install pdfplumber  # or PyPDF2
pip install sentence-transformers  # for SBERT support (optional)
pip install matplotlib numpy pandas
pip install tqdm
```

### NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Workflow

### 1. Download Corpus
```bash
python download_corpus_arxiv.py --author "Edward Witten"
```

### 2. Run Plagiarism Detection
```bash
# For arXiv corpus
python codetorun.py --mode detection

# For PAN corpus
python eval_plagiarism.py
```

### 3. Analyze Results
```bash
python analyze_plagiarism_stats.py
```

## Configuration

### Key Parameters

**Similarity Thresholds:**
- `similarity_threshold`: BM25 threshold (default: 0.5)
- `sbert_threshold`: SBERT semantic similarity threshold (default: 0.65)
- `jaccard_threshold`: Jaccard lexical similarity threshold (default: 0.4)

**Chunking:**
- `chunk_size`: Words per chunk (default: 200)
- `overlap`: Overlapping words between chunks (default: 50)
- `min_word_overlap_ratio`: Minimum word overlap ratio (default: 0.3)

**Ranking:**
- `top_k`: Number of top matches to consider per chunk (default: 5-20)
- `use_sbert`: Enable SBERT semantic similarity (default: True)
- `use_jaccard`: Enable Jaccard similarity (default: False)
- `hybrid_mode`: Combine multiple ranking methods (default: False)

## Output Files

### Detection Results
- `plagiarism_report.txt`: Human-readable report with YES/NO answers and examples
- `plagiarism_chunks.json`: Detailed JSON with all plagiarism chunks
- `plagiarism_statistics.json`: Statistics per document
- `plagiarism_statistics_report.txt`: Text report of statistics
- `*.png`: Visualization files (histograms, box plots, scatter plots)

### Query Results
- `query_results_[paper_name]/query_results.json`: JSON results
- `query_results_[paper_name]/query_report.txt`: Human-readable report

## Dependencies

### Required
- Python 3.7+
- nltk
- numpy
- matplotlib
- pandas

### Optional
- sentence-transformers (for SBERT semantic similarity)
- pdfplumber or PyPDF2 (for PDF text extraction)
- feedparser (for arXiv API)

## Notes

- The system uses chunk-based detection to identify plagiarism in specific sections
- Multiple ranking methods can be combined for better accuracy
- SBERT provides semantic similarity detection (catches paraphrasing)
- BM25 provides lexical similarity detection (catches copy-paste)
- Results include both similarity scores and example passages
- The system filters out very short chunks (< 20 words) to reduce noise

## License

This project is part of a course assignment (CS 650).

## Author

Plagiarism Detection System for CS 650
