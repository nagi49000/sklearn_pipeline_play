import random
from faker import Faker
from typing import (
    Set,
    Iterable,
    List,
    Dict,
    Callable
)
from faker.providers import internet
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass


class FakeDataGenerator():
    def get_hostnames(
            self,
            number_of_records: int,
            levels: List[int]=[2],
            seed: int=None
    ) -> Iterable[str]:
        """ number_of_records - number of domain names to create
            levels - list of depths that hostnames can have
            seed - optional seed for repeatable results
        """
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        fake = Faker()
        fake.add_provider(internet)
        for _ in range(number_of_records):
            yield fake.hostname(levels=random.choice(levels))


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, split_char: str="."):
        """ split_char - character over which to split strings """
        self.split_char = split_char

    def fit(self, X, y=None):
        """ trivial method; returns self """
        return self

    def transform(self, X: Iterable[str], y: None=None) -> Iterable[List[str]]:
        """ convert str to list<str> split over split_char """
        for s in X:
            # split the string, dropping zero length strings
            split_str = [x for x in s.split(self.split_char) if x]
            yield split_str


class StopWordRemover(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words: Set[str]=None):
        """ stop_words - tokens to drop from supplied tokens """
        self.stop_words = set() if stop_words is None else stop_words

    def fit(self, X, y=None):
        """ trivial method; returns self """
        return self

    def transform(self, X: Iterable[List[str]], y: None=None) -> Iterable[List[str]]:
        """ filter a list<str> for stop words """
        for s in X:
            filtered_str = [x for x in s if x not in self.stop_words]
            yield filtered_str


@dataclass
class TokensWithReferences:
    tokens: List[str]
    references: List[str]


def get_references(tokens: List[str]) -> List[str]:
    """ based on the supplied tokens, return a list of relevant reference datasets """
    references = []
    if tokens:
        references.append("country")
        if tokens[-1] == "edu" or "ac" in tokens:
            references.append("uni")
    return references


class ReferenceAdder(BaseEstimator, TransformerMixin):
    def __init__(self, get_references_function: Callable=None):
        """
            get_references_function - function that takes a list of tokens, and
                                      returns a list of reference datasets
        """
        def empty_fn(_):
            return []
        self.get_references_function = empty_fn if get_references_function is None else get_references_function

    def fit(self, X, y=None):
        """ trivial method; returns self """
        return self

    def transform(self, X: Iterable[List[str]], y: None=None) -> Iterable[TokensWithReferences]:
        for s in X:
            yield TokensWithReferences(s, self.get_references_function(s))


class ReferenceResolver(BaseEstimator, TransformerMixin):
    def __init__(self, reference_datasets: Dict[str, Dict[str, str]]):
        """
            reference_datasets - key is a str, and value is a reference dataset.
                                 A reference dataset is a dictionary acting as a lookup
                                 of words to lemmatized words
        """
        self.reference_datasets = reference_datasets

    def fit(self, X, y=None):
        """ trivial method; returns self """
        return self

    def transform(self, X: Iterable[TokensWithReferences], y: None=None) -> Iterable[List[str]]:
        for s in X:
            resolved = []
            for ref in s.references:
                this_ref_ds = self.reference_datasets.get(ref, {})
                this_resolved = [this_ref_ds.get(x, None) for x in s.tokens]
                this_resolved = [x for x in this_resolved if x is not None]
                resolved += this_resolved
            yield resolved


pipeline = Pipeline(steps=[
    ("tokenize", Tokenizer()),
    ("stopwords", StopWordRemover()),
    ("reference_enrichment", ReferenceAdder(get_references)),
    ("reference_resolve", ReferenceResolver({}))
])
