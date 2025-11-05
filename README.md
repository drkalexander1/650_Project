# Corpus Download Scripts for Plagiarism Detection Project

Scripts to download papers from PubMed/PMC or arXiv for building a corpus for in-group plagiarism detection in academic research networks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For PubMed/PMC scripts, update the email address in `download_corpus.py` (line 24):
   - Change `Entrez.email = "your.email@example.com"` to your email address
   - NCBI requires a valid email for API access

## Available Scripts

### 1. PubMed/PMC Download (`download_corpus.py`)

Downloads papers from PubMed/PMC (biomedical and life sciences).

**Features:**
- Searches PubMed database
- Downloads full-text XML from PubMed Central (PMC)
- Extracts text from XML format
- Filters for open access papers only

**Usage:**
```bash
python download_corpus.py --author "Kao CH" --limit 5
```

### 2. arXiv Download (`download_corpus_arxiv.py`)

Downloads papers from arXiv (physics, mathematics, computer science, etc.).

**Features:**
- Searches arXiv database
- Downloads PDF files
- Extracts text from PDF format
- All arXiv papers are open access

**Usage:**
```bash
python download_corpus_arxiv.py --author "Kao CH" --limit 5
```

## Usage

### Basic Usage

**PubMed/PMC:**
```bash
python download_corpus.py
```

**arXiv:**
```bash
python download_corpus_arxiv.py
```

### Test Mode (Recommended First)

Test with a small number of papers:

```bash
python download_corpus.py --limit 5
python download_corpus_arxiv.py --limit 5
```

### Command Line Options

Both scripts support the following options:

- `--author`, `-a`: Author name (default: "Kao CH")
  ```bash
  python download_corpus.py --author "Smith J"
  ```

- `--limit`, `-l`: Limit number of papers to process (for testing)
  ```bash
  python download_corpus.py --limit 10
  ```

- `--output`, `-o`: Output directory (default: "corpus")
  ```bash
  python download_corpus.py --output my_corpus
  ```

- `--min-words`: Minimum word count to consider as full-text (default: 500)
  ```bash
  python download_corpus.py --min-words 1000
  ```

- `--save-abstracts`: Save abstract-only papers in a separate folder instead of skipping them
  ```bash
  python download_corpus.py --save-abstracts
  ```

**PubMed/PMC specific:**
- `--all-papers`: Include all papers (not just open access). Default: open access only
  ```bash
  python download_corpus.py --all-papers
  ```

## Output Structure

Both scripts create the following directory structure:

**PubMed/PMC:**
```
corpus/
├── xml/              # Raw PMC XML files
├── text/             # Full-text articles (plain text)
├── metadata/         # JSON files with paper metadata
├── abstracts_only/   # Abstract-only papers (if --save-abstracts used)
└── corpus_summary.json  # Summary of all downloaded papers
```

**arXiv:**
```
corpus/
├── pdf/              # Raw PDF files
├── text/             # Full-text articles (plain text)
├── metadata/         # JSON files with paper metadata
├── abstracts_only/   # Abstract-only papers (if --save-abstracts used)
└── corpus_summary.json  # Summary of all downloaded papers
```

## Features

- **Open Access Filter**: Only downloads papers available for free
- **Full-Text Only**: Skips papers without full-text content (configurable threshold)
- **Rate Limiting**: Respects API rate limits
- **Error Handling**: Gracefully handles failed downloads and missing content
- **Metadata Extraction**: Captures title, authors, journal/venue, year, abstract, etc.

## Example Output

```
Starting corpus download for author: Kao CH
Output directory: /path/to/corpus
Source: arXiv
Full-text threshold: Minimum 500 words
Papers without full-text will be skipped

Searching arXiv for author: Kao CH
Found 25 papers in arXiv

Downloading articles...
[1/25] Processing arXiv ID: 2312.12345
  Title: Example Paper Title...
  Downloading PDF from arXiv...
  Saved full-text: corpus/text/2312.12345.txt (2847 words)
...
```

## Notes

- **PubMed/PMC**: Script respects NCBI rate limits (~3 requests/second)
- **arXiv**: Script includes 1-second delay between requests
- Papers with publisher restrictions (PubMed) or extraction issues may only have abstracts available
- Full-text papers are saved to `text/` directory
- Abstract-only papers are skipped by default (use `--save-abstracts` to keep them)

## Project Context

This script is part of a project for detecting in-group plagiarism in academic research networks. See `proposal.txt` for more details about the overall project goals.


