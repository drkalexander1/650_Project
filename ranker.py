"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from indexing import InvertedIndex
from sentence_transformers import CrossEncoder

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document (Not needed for HW1)
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query (Hint: Also apply stopwords filtering to the tokenized query)
        query_tokens = self.tokenize(query)
        if self.stopwords:
            query_tokens = [token if token not in self.stopwords else None for token in query_tokens]

        # Count query word frequencies
        query_word_counts = Counter(query_tokens)

        # 2.1 For each token in the tokenized query, find out all documents that contain it and counting its frequency within each document.
        # Hint 1: To understand why we need the info above, pay attention to docid and doc_word_counts,
        #    located in the score() function within the RelevanceScorer class
        # Hint 2: defaultdict(Counter) works well in this case, where we store {docids : {query_tokens : counts}},
        #         or you may choose other approaches
        doc_word_counts = defaultdict(Counter)
        relevant_docs = set()

        for token in query_tokens:
            if token is None:
                continue
            postings = self.index.get_postings(token)
            for doc_id, term_freq in postings:
                doc_word_counts[doc_id][token] = term_freq
                relevant_docs.add(doc_id)

        # 2.2 Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        #        for each relevant document determined in 2.1
        scores = []
        for doc_id in relevant_docs:
            score = self.scorer.score(doc_id, doc_word_counts[doc_id], query_word_counts)
            scores.append((doc_id, score))

        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        # Document IDs now match between internal and external (no conversion needed)
        return sorted([(doc_id, score) for doc_id, score in scores], key=lambda x: x[1], reverse=True)


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document that are also found in the query, 
                                and their frequencies within the document.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        dot_product = 0
        for term, query_freq in query_word_counts.items():
            if term in doc_word_counts:
                dot_product += doc_word_counts[term] * query_freq

        # 2. Return the score as integer for WordCountCosineSimilarity
        return dot_product


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters
        self.verbose = parameters.get('verbose', False)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_metadata = self.index.get_doc_metadata(docid)
        doc_length = doc_metadata.get('length', 0)
        stats = self.index.get_statistics()
        total_tokens = stats.get('total_token_count', 1)

        self._debug(f"\n=== DIRICHLET LM DEBUG for docid {docid} ===")
        self._debug(f"doc_length: {doc_length}")
        self._debug(f"total_tokens: {total_tokens}")
        self._debug(f"query_word_counts: {query_word_counts}")
        self._debug(f"doc_word_counts: {doc_word_counts}")

        # 2. Compute additional terms to use in algorithm
        mu = self.parameters['mu']
        self._debug(f"mu: {mu}")
        # Dirichlet smoothing formula: P(w|d) = qtf * log(1+ tf_doc / mu / P(w|C)) + |q| * log(mu /(|d| + mu))

        # 3. For all query_parts, compute score
        score = 0.0
        for term, query_freq in query_word_counts.items():
            if term in doc_word_counts:
                # Query term frequency (QTF): just use the query frequency
                qtf = query_freq

                # Term frequency in document
                tf_doc = doc_word_counts[term]

                # Collection frequency of term
                term_metadata = self.index.get_term_metadata(term)
                cf = term_metadata.get('term_count', 0)

                self._debug(f"\n--- Term: {term} ---")
                self._debug(f"qtf: {qtf}")
                self._debug(f"tf_doc: {tf_doc}")
                self._debug(f"cf: {cf}")

                # Dirichlet smoothing: P(w|d) = qtf * log(1+ tf_doc / mu / P(w|C)) + |q| * log(mu /(|d| + mu))
                # where P(w|C) is the probability of the term in the collection
                p_w_c = cf / total_tokens if total_tokens > 0 else 0
                self._debug(f"p_w_c = {cf} / {total_tokens} = {p_w_c}")

                p_w_d = qtf * np.log(1 + tf_doc / (mu * p_w_c))
                self._debug(f"p_w_d = {qtf} * log(1 + {tf_doc} / ({mu} * {p_w_c})) = {p_w_d}")

                score += p_w_d
                self._debug(f"running_score: {score}")

        # Add normalized document length term outside of the loop
        length_term = len(query_word_counts) * np.log(mu / (doc_length + mu))
        self._debug(f"\nlength_term = {len(query_word_counts)} * log({mu} / ({doc_length} + {mu})) = {length_term}")
        score += length_term

        # 4. Return the score
        self._debug(f"FINAL DIRICHLET LM SCORE: {score}")
        return score


# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters.get('b', 0.65)
        self.k1 = parameters.get('k1', 1.5)
        self.k3 = parameters.get('k3', 8)
        self.verbose = parameters.get('verbose', False)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_metadata = self.index.get_doc_metadata(docid)
        doc_length = doc_metadata.get('length', 0)
        stats = self.index.get_statistics()
        avg_doc_length = stats.get('mean_document_length', 1)
        total_docs = stats.get('number_of_documents', 1)

        self._debug(f"\n=== BM25 DEBUG for docid {docid} ===")
        self._debug(f"doc_length: {doc_length}")
        self._debug(f"avg_doc_length: {avg_doc_length}")
        self._debug(f"total_docs: {total_docs}")
        self._debug(f"b: {self.b}, k1: {self.k1}, k3: {self.k3}")
        self._debug(f"query_word_counts: {query_word_counts}")
        self._debug(f"doc_word_counts: {doc_word_counts}")

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        # (This is handled in the main loop below)

        # 3. For all query parts, compute the TF and IDF to get a score
        score = 0.0
        for term, query_freq in query_word_counts.items():
            if term in doc_word_counts:
                # Term frequency in document
                tf_doc = doc_word_counts[term]

                # Document frequency of term
                term_metadata = self.index.get_term_metadata(term)
                df = term_metadata.get('doc_frequency', 0)

                self._debug(f"\n--- Term: {term} ---")
                self._debug(f"tf_doc: {tf_doc}")
                self._debug(f"df: {df}")
                self._debug(f"query_freq: {query_freq}")

                # BM25 formula: IDF * (tf_doc * (k1 + 1)) / (tf_doc + k1 * (1 - b + b * (|d| / avg_doc_length))) * qtf
                # where IDF = log((N - df + 0.5) / (df + 0.5))

                # IDF calculation
                idf = np.log((total_docs - df + 0.5) / (df + 0.5))
                self._debug(f"idf = log(({total_docs} - {df} + 0.5) / ({df} + 0.5)) = {idf}")

                # TF calculation with length normalization
                length_norm_factor = 1 - self.b + self.b * (doc_length / avg_doc_length)
                self._debug(f"length_norm_factor = 1 - {self.b} + {self.b} * ({doc_length} / {avg_doc_length}) = {length_norm_factor}")

                tf_norm = (tf_doc * (self.k1 + 1)) / (tf_doc + self.k1 * length_norm_factor)
                self._debug(f"tf_norm = ({tf_doc} * ({self.k1} + 1)) / ({tf_doc} + {self.k1} * {length_norm_factor}) = {tf_norm}")

                # Query term frequency component
                qtf = ((self.k3 + 1) * query_freq) / (self.k3 + query_freq)
                self._debug(f"qtf = (({self.k3} + 1) * {query_freq}) / ({self.k3} + {query_freq}) = {qtf}")

                # Add to total score
                term_score = idf * tf_norm * qtf
                self._debug(f"term_score = {idf} * {tf_norm} * {qtf} = {term_score}")
                score += term_score
                self._debug(f"running_score: {score}")

        # 4. Return score
        self._debug(f"FINAL BM25 SCORE: {score}")
        return score


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters.get('b', 0.2)
        self.verbose = parameters.get('verbose', False)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_metadata = self.index.get_doc_metadata(docid)
        doc_length = doc_metadata.get('length', 0)
        stats = self.index.get_statistics()
        avg_doc_length = stats.get('mean_document_length', 1)
        total_docs = stats.get('number_of_documents', 1)

        self._debug(f"\n=== PIVOTED NORMALIZATION DEBUG for docid {docid} ===")
        self._debug(f"doc_length: {doc_length}")
        self._debug(f"avg_doc_length: {avg_doc_length}")
        self._debug(f"total_docs: {total_docs}")
        self._debug(f"b: {self.b}")
        self._debug(f"query_word_counts: {query_word_counts}")
        self._debug(f"doc_word_counts: {doc_word_counts}")

        # 2. Compute additional terms to use in algorithm
        # Pivoted normalization formula: qtf * 1+log(1+log(tf_doc))/(1-b+b*|d|/avg_doc_length) * log((total_docs+1)/df)

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        score = 0.0
        for term, query_freq in query_word_counts.items():
            if term in doc_word_counts:
                # Term frequency in document
                tf_doc = doc_word_counts[term]

                # Document frequency of term
                term_metadata = self.index.get_term_metadata(term)
                df = term_metadata.get('doc_frequency', 0)

                self._debug(f"\n--- Term: {term} ---")
                self._debug(f"tf_doc: {tf_doc}")
                self._debug(f"df: {df}")
                self._debug(f"query_freq: {query_freq}")

                # Pivoted normalization formula:
                # score += query_freq * 1+log(1+log(tf_doc))/(1-b+b*|d|/avg_doc_length) * log((total_docs+1)/df)

                # TF component: normalized by document length
                tf_component = np.log(tf_doc) + 1 if tf_doc > 0 else 0
                self._debug(f"tf_component = log({tf_doc}) + 1 = {tf_component}")

                # IDF component: log((total_docs +1) / df)
                idf_component = np.log((total_docs + 1) / df) if df > 0 else 0
                self._debug(f"idf_component = log(({total_docs} + 1) / {df}) = {idf_component}")

                # Length normalization: 1 - b + b * (|d| / avg_doc_length)
                length_norm = 1 - self.b + self.b * (doc_length / avg_doc_length)
                self._debug(f"length_norm = 1 - {self.b} + {self.b} * ({doc_length} / {avg_doc_length}) = {length_norm}")

                # Query term frequency (QTF): just use the query frequency
                qtf = query_freq
                self._debug(f"qtf: {qtf}")

                # Add to total score
                term_score = (1+np.log(tf_component)) * idf_component / length_norm * qtf
                self._debug(f"term_score = (1+log({tf_component})) * {idf_component} / {length_norm} * {qtf} = {term_score}")
                score += term_score
                self._debug(f"running_score: {score}")

        # 4. Return the score
        self._debug(f"FINAL PIVOTED NORMALIZATION SCORE: {score}")
        return score


# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        stats = self.index.get_statistics()
        total_docs = stats.get('number_of_documents', 1)

        # 2. Compute additional terms to use in algorithm
        # TF-IDF uses: tf * idf where tf = log(term_count_in_document + 1)
        # and idf = log(total_number_of_documents / document_with_term) + 1

        # 3. For all query parts, compute the TF and IDF to get a score
        score = 0.0
        for term, query_freq in query_word_counts.items():
            if term in doc_word_counts:
                # Term frequency in document
                tf_doc = doc_word_counts[term]

                # Document frequency of term
                term_metadata = self.index.get_term_metadata(term)
                df = term_metadata.get('doc_frequency', 0)

                # TF-IDF formula: log(tf_d+1) * log(total_docs / df)+1

                # TF component: normalized by document length
                tf_component = np.log(tf_doc + 1)

                # IDF component: log(total_docs / df) + 1
                idf_component = np.log(total_docs / df)+1 if df > 0 else 0

                # Add to total score
                score += tf_component * idf_component

        # 4. Return the score
        return score
