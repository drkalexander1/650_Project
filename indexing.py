'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''
import os
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm
from preprocessing import Tokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex and PositionalIndex
    BasicInvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.

        Note: The following variables are defined to help you store some summary info about your document collection
                for a quick look-up.
              You may also define more variables and/or keys as you see fit.
        Variables:
            statistics: A dictionary, which is the central statistics of the index.
                        Some keys include:
                statistics['vocab']: A counter which keeps track of the token count
                statistics['unique_token_count']: how many unique terms are in the index
                statistics['total_token_count']: how many total tokens are indexed including filterd tokens),
                    i.e., the sum of the lengths of all documents
                statistics['stored_total_token_count']: how many total tokens are indexed excluding filtered tokens
                statistics['number_of_documents']: the number of documents indexed
                statistics['mean_document_length']: the mean number of tokens in a document (including filter tokens)
                ...
                (Add more keys to the statistics dictionary as you see fit)

            vocabulary: A set of distinct words that have appeared in the collection
            document_metadata: A dictionary, which keeps track of some important metadata for each document.
                               Assume that we have a document called 'doc1', some keys include:
                document_metadata['doc1']['unique_tokens']: How many unique tokens are in the document (among those not-filtered)
                document_metadata['doc1']['length']: How long the document is in terms of tokens (including those filtered) 
                ...
                (Add more keys to the document_metadata dictionary as you see fit)
            index: A dictionary of class defaultdict, its implemention depends on whether we are using 
                            BasicInvertedIndex or PositionalIndex.
                    BasicInvertedIndex: Store the mapping of terms to their postings
                    PositionalIndex: Each term keeps track of documents and positions of the terms occurring in the document

        Example:
            document1 = ['This', 'is' ,'a', 'dog', None]
            document2 = [None, 'This', 'is', 'a', 'cat']

            statistics = {
                'vocab'                     : Counter({'This': 2, 'is': 2, 'a' : 2, 'dog' : 1, 'cat': 1}),
                'unique_token_count'        : 5,
                'total_token_count'         : 10,
                'stored_total_token_count'  : 8,
                'number_of_documents'       : 2,
                'mean_document_length'      : 5
            }

            vocabulary = {'This', 'is', 'a', 'cat', 'dog'}

            document_metadata = {
                'document1': {'unique_tokens': 4, 'length': 5},
                'document2': {'unique_tokens': 4, 'length': 5}
            }

            If BasicInvertedIndex, we store 'term': ((docid, count))
            index = {
                'This': (('document1', 1), ('document2', 1)),
                'is': (('document1', 1), ('document2', 1)),
                'a': (('document1', 1), ('document2', 1)),
                'dog': (('document1', 1))
                'cat': (('document2', 1))
            }
            If PositionalIndex, we store 'term': ((docid, count, position))
            index = {
                'This': (('document1', 1, 0), ('document2', 1, 1)),
                'is': (('document1', 1, 1), ('document2', 1, 2)),
                'a': (('document1', 1, 2), ('document2', 1, 3)),
                'dog': (('document1', 1, 3))
                'cat': (('document2', 1, 4))
            }

        """
        # Define necessary variables
        # We will use them later when we implement the below methods in BasicInvertedIndex and PositionalIndex classes
        self.statistics = {}
        self.statistics['vocab'] = Counter()
        self.vocabulary = set()
        self.document_metadata = {}
        self.index = defaultdict(list)

    # NOTE: The following functions have to be implemented in the two inherited classes and NOT in this class

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Implement this to add documents to the index
        raise NotImplementedError

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Implement to fetch a particular document stored in metadata
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Implement to fetch a particular term stored in metadata
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic

        # Hint: Statistics only need recomputation if the index has been modified since the last calculation.
        #       This ensure substantial performance improvements.
        raise NotImplementedError

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self._stats_computed = False  # Flag to track if statistics need recomputation

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # Count tokens (including None values for filtered tokens)
        total_tokens = len(tokens)

        # Process tokens in a single pass to avoid multiple iterations
        term_counts = Counter()
        unique_tokens = set()
        for token in tokens:
            if token is not None:
                term_counts[token] += 1
                unique_tokens.add(token)

        # Update document metadata
        self.document_metadata[docid] = {
            'unique_tokens': len(unique_tokens),
            'length': total_tokens
        }

        # Update vocabulary and statistics in one operation
        self.vocabulary.update(unique_tokens)
        self.statistics['vocab'].update(term_counts)

        # Update the inverted index
        for term, count in term_counts.items():
            self.index[term].append((docid, count))

        # Mark statistics as needing recomputation
        self._stats_computed = False

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        if docid not in self.document_metadata:
            return  # Document doesn't exist

        # Remove from document metadata
        del self.document_metadata[docid]

        # Remove from index and update vocabulary in one pass
        for term, postings in list(self.index.items()):  # Use list() to avoid modification during iteration
            # Filter out postings for this document
            new_postings = [posting for posting in postings if posting[0] != docid]

            if new_postings:
                self.index[term] = new_postings
            else:
                # Remove immediately instead of collecting for later
                del self.index[term]
                self.vocabulary.discard(term)
                self.statistics['vocab'].pop(term, None)

        # Mark statistics as needing recomputation
        self._stats_computed = False

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        postings = self.get_postings(term)
        term_count = self.statistics['vocab'].get(term, 0)
        doc_frequency = len(postings)

        return {
            'term_count': term_count,
            'doc_frequency': doc_frequency
        }

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include AT LEAST the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        if not self._stats_computed:
            # Calculate statistics
            unique_token_count = len(self.vocabulary)
            total_token_count = sum(metadata['length'] for metadata in self.document_metadata.values())
            stored_total_token_count = sum(self.statistics['vocab'].values())
            number_of_documents = len(self.document_metadata)
            mean_document_length = total_token_count / number_of_documents if number_of_documents > 0 else 0

            # Update statistics
            self.statistics.update({
                'unique_token_count': unique_token_count,
                'total_token_count': total_token_count,
                'stored_total_token_count': stored_total_token_count,
                'number_of_documents': number_of_documents,
                'mean_document_length': mean_document_length
            })

            self._stats_computed = True

        return self.statistics

    def save(self, index_directory_name: str = 'tmp') -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # Create directory if it doesn't exist
        os.makedirs(index_directory_name, exist_ok=True)

        # Save index
        index_file = os.path.join(index_directory_name, 'index.json')
        with open(index_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            index_dict = {term: postings for term, postings in self.index.items()}
            json.dump(index_dict, f)

        # Save statistics
        stats_file = os.path.join(index_directory_name, 'statistics.json')
        with open(stats_file, 'w') as f:
            # Convert Counter to dict for JSON serialization
            stats_dict = dict(self.statistics)
            stats_dict['vocab'] = dict(stats_dict['vocab'])
            json.dump(stats_dict, f)

        # Save vocabulary
        vocab_file = os.path.join(index_directory_name, 'vocabulary.json')
        with open(vocab_file, 'w') as f:
            json.dump(list(self.vocabulary), f)

        # Save document metadata
        metadata_file = os.path.join(index_directory_name, 'document_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.document_metadata, f)

    def load(self, index_directory_name: str = 'tmp') -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # Load index
        index_file = os.path.join(index_directory_name, 'index.json')
        with open(index_file, 'r') as f:
            index_dict = json.load(f)
            self.index = defaultdict(list, index_dict)

        # Load statistics
        stats_file = os.path.join(index_directory_name, 'statistics.json')
        with open(stats_file, 'r') as f:
            stats_dict = json.load(f)
            self.statistics = stats_dict
            # Convert vocab back to Counter
            self.statistics['vocab'] = Counter(stats_dict['vocab'])

        # Load vocabulary
        vocab_file = os.path.join(index_directory_name, 'vocabulary.json')
        with open(vocab_file, 'r') as f:
            self.vocabulary = set(json.load(f))

        # Load document metadata
        metadata_file = os.path.join(index_directory_name, 'document_metadata.json')
        with open(metadata_file, 'r') as f:
            # JSON converts integer keys to strings, so we need to convert them back
            loaded_metadata = json.load(f)
            self.document_metadata = {int(docid): metadata for docid, metadata in loaded_metadata.items()}

        # Force recomputation of statistics to ensure all required fields are present
        # This is important because loaded statistics might be incomplete or outdated
        self._stats_computed = False
        self.get_statistics()  # This will compute and cache the statistics

class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalInvertedIndex'

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the positional index and update the index's metadata.
        This stores positional information for each term occurrence.

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # Count tokens (including None values for filtered tokens)
        total_tokens = len(tokens)
        valid_tokens = [token for token in tokens if token is not None]
        unique_tokens = set(valid_tokens)

        # Update document metadata
        self.document_metadata[docid] = {
            'unique_tokens': len(unique_tokens),
            'length': total_tokens
        }

        # Update vocabulary and statistics
        self.vocabulary.update(valid_tokens)

        # Count term frequencies and track positions
        term_positions = defaultdict(list)
        for position, token in enumerate(tokens):
            if token is not None:
                term_positions[token].append(position)

        # Update the inverted index with positional information
        for term, positions in term_positions.items():
            count = len(positions)
            self.index[term].append((docid, count, positions))

        # Update vocabulary counter
        self.statistics['vocab'].update(valid_tokens)

        # Mark statistics as needing recomputation
        self._stats_computed = False

    def get_postings(self, term: str) -> list[tuple[int, int, list[int]]]:
        """
        Returns the list of postings for a positional index, which contains documents, term counts, and positions.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing (docid, count, positions) for documents containing the term
        """
        return self.index.get(term, [])


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, id_key: str | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
                Setting a value that is less than 1 (e.g. 0 or negative) will result in all documents being indexed.

        Returns:
            An inverted index
        '''
        # Create the appropriate index type
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Determine file opening method based on extension
        if dataset_path.endswith('.gz'):
            file_opener = gzip.open
            mode = 'rt'
        else:
            file_opener = open
            mode = 'r'

        # First pass: Calculate word frequencies if minimum_word_frequency > 0
        word_frequencies = Counter()
        if minimum_word_frequency > 0:
            print("First pass: Calculating word frequencies...")
            with file_opener(dataset_path, mode) as f:
                doc_count = 0
                for line in tqdm(f, desc="Calculating frequencies"):
                    if max_docs > 0 and doc_count >= max_docs:
                        break

                    try:
                        doc = json.loads(line.strip())
                        if text_key in doc:
                            # Tokenize without filtering to get all word frequencies
                            tokens = document_preprocessor.tokenize(doc[text_key])
                            # Count all non-None tokens
                            valid_tokens = [token for token in tokens if token is not None]
                            word_frequencies.update(valid_tokens)
                            doc_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

            print(f"Processed {doc_count} documents for frequency calculation")

        # Determine which words to filter out
        words_to_filter = set()
        if stopwords:
            words_to_filter.update(stopwords)

        if minimum_word_frequency > 0:
            low_freq_words = {word for word, count in word_frequencies.items() 
                             if count < minimum_word_frequency}
            words_to_filter.update(low_freq_words)
            print(f"Filtering out {len(low_freq_words)} low-frequency words")

        print(f"Total words to filter: {len(words_to_filter)}")

        # Second pass: Process documents and build index
        print("Second pass: Building index...")
        with file_opener(dataset_path, mode) as f:
            doc_count = 0
            for line in tqdm(f, desc="Building index"):
                if max_docs > 0 and doc_count >= max_docs:
                    break

                try:
                    doc = json.loads(line.strip())
                    if text_key in doc:
                        # Tokenize the document
                        tokens = document_preprocessor.tokenize(doc[text_key])

                        # Filter out stopwords and low-frequency words
                        filtered_tokens = []
                        for token in tokens:
                            if token is None or token in words_to_filter:
                                filtered_tokens.append(None)
                            else:
                                filtered_tokens.append(token)

                        # Resolve document id from JSON
                        resolved_doc_id = None
                        if id_key is not None:
                            # Require the specified id_key; skip if missing or invalid
                            if isinstance(doc, dict) and id_key in doc and doc[id_key] is not None:
                                try:
                                    resolved_doc_id = int(doc[id_key])
                                except (ValueError, TypeError):
                                    # Skip documents with non-integer IDs
                                    continue
                            else:
                                # Skip documents without the required id_key
                                continue
                        else:
                            # Auto-detect common id keys; otherwise fall back to sequential ids
                            if isinstance(doc, dict):
                                for candidate_key in ['curid', 'pageid', 'id', 'docid', '_id']:
                                    if candidate_key in doc and doc[candidate_key] is not None:
                                        try:
                                            resolved_doc_id = int(doc[candidate_key])
                                            break
                                        except (ValueError, TypeError):
                                            continue
                            if resolved_doc_id is None:
                                resolved_doc_id = doc_count

                        # Skip duplicates to avoid overwriting metadata/postings
                        if resolved_doc_id in index.document_metadata:
                            continue

                        # Add document to index
                        index.add_doc(resolved_doc_id, filtered_tokens)
                        doc_count += 1

                except (json.JSONDecodeError, KeyError):
                    continue

        print(f"Indexed {doc_count} documents")
        print(f"Vocabulary size: {len(index.vocabulary)}")

        return index
