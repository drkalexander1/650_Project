# Corpus Download Script for Plagiarism Detection Project

Script to download papers from PubMed/PMC for building a corpus for in-group plagiarism detection in academic research networks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update the email address in `download_corpus.py` (line 24):
   - Change `Entrez.email = "your.email@example.com"` to your email address
   - NCBI requires a valid email for API access

## Usage

### Basic Usage

Download papers for a specific author (default: "Kao CH"):

```bash
python download_corpus.py
```

### Test Mode (Recommended First)

Test with a small number of papers:

```bash
python download_corpus.py --limit 5
```

### Command Line Options

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

- `--all-papers`: Include all papers (not just open access). Default: open access only
  ```bash
  python download_corpus.py --all-papers
  ```

## Output Structure

The script creates the following directory structure:

```
corpus/
├── xml/              # Raw PMC XML files
├── text/             # Full-text articles (plain text)
├── metadata/         # JSON files with paper metadata
├── abstracts_only/   # Abstract-only papers (if --save-abstracts used)
└── corpus_summary.json  # Summary of all downloaded papers
```

## Features

- **Open Access Filter**: By default, only downloads papers available in PMC (open access)
- **Full-Text Only**: Skips papers without full-text content (configurable threshold)
- **Rate Limiting**: Respects NCBI API rate limits
- **Error Handling**: Gracefully handles failed downloads and missing content
- **Metadata Extraction**: Captures title, authors, journal, year, abstract, etc.

## Example Output

```
Starting corpus download for author: Kao CH
Output directory: /path/to/corpus
Mode: Open access papers only (will skip papers without PMC access)
Full-text threshold: Minimum 500 words
Papers without full-text will be skipped

Searching PubMed for author: Kao CH
  Filtering for open access papers (available in PMC)
Found 157 papers in PubMed (open access only)

Converting 157 PMIDs to PMCIDs...
Found 157 papers available in PMC

Downloading articles...
[1/157] Processing PMID: 40948537
  Downloading full-text from PMC: 12432558
  Saved full-text: corpus/text/PMC12432558.txt (2847 words)
...
```

## Notes

- The script respects NCBI rate limits (~3 requests/second)
- Papers with publisher restrictions may only have abstracts available
- Full-text papers are saved to `text/` directory
- Abstract-only papers are skipped by default (use `--save-abstracts` to keep them)

## Project Context

This script is part of a project for detecting in-group plagiarism in academic research networks. See `proposal.txt` for more details about the overall project goals.

