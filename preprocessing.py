"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
from nltk.tokenize import RegexpTokenizer
import nltk
import spacy
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download NLTK stopwords data if not already present
try:
    nltk.download('stopwords', quiet=True)
except Exception:
    pass  # If download fails, will handle it when loading stopwords


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
        self.sorted_mwes = (
            sorted(self.multiword_expressions, key=lambda x: len(x.split()), reverse=True)
            if self.multiword_expressions else []
        )

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition

        Examples:
            If lowercase, "Taylor" "Swift" -> "taylor" "swift"
            If "Taylor Swift" in multiword_expressions, "Taylor" "Swift" -> "Taylor Swift"
        """
        output_tokens = input_tokens.copy()
        if self.lowercase:
            output_tokens = [token.lower() for token in output_tokens]

        return output_tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SampleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        """This is a dummy tokenizer.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        return ['token_1', 'token_2', 'token_3']  # This is not correct; it is just a placeholder.


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        The Natural Language Toolkit (NLTK) is a Python package for natural language processing.
        To learn more, visit https://pypi.org/project/nltk/

        Installation Instructions:
            Please visit https://spacy.io/usage
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U nltk

        Your tasks:
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        self.token_regex = token_regex
        # TODO: Initialize the NLTK's RegexpTokenizer
        self.tokenizer = RegexpTokenizer(self.token_regex)
        # Set Multiword Expressions
        self.multiword_expressions = multiword_expressions

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens using the postprocess function
        # Tokenize using NLTK's RegexpTokenizer
        tokens = self.tokenizer.tokenize(text)

        # Apply postprocessing (lowercasing and multi-word expressions)
        return self.postprocess(tokens)


def load_nltk_stopwords(language: str = 'english') -> set[str]:
    """
    Load stopwords from NLTK's stopwords corpus.
    
    Args:
        language: Language code for stopwords (default: 'english')
                  Options include: 'english', 'spanish', 'french', 'german', etc.
    
    Returns:
        Set of stopwords as lowercase strings
    
    Raises:
        LookupError: If the stopwords corpus is not available and cannot be downloaded
    """
    try:
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words(language))
        return stopwords_set
    except LookupError:
        # Try to download the stopwords corpus
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stopwords_set = set(stopwords.words(language))
            return stopwords_set
        except Exception as e:
            raise LookupError(
                f"Failed to load NLTK stopwords for language '{language}'. "
                f"Please ensure NLTK is installed and the stopwords corpus is available. "
                f"Error: {e}"
            )
