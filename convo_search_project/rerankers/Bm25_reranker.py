from pygaggle.rerank.base import Reranker, Query, Text
from pyserini.index import IndexReader
from typing import List

'''
class Bm25Reranker(Reranker):
    def __init__(self):
        self.index_reader = IndexReader.from_prebuilt_index('cast19')

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
'''

from collections import Counter
from copy import deepcopy
from typing import List
import math

from pyserini.analysis import get_lucene_analyzer, Analyzer
from pyserini.index import IndexReader
import numpy as np


class Bm25Reranker(Reranker):
    def __init__(self, index_reader, similarity):
        self.index_reader = index_reader
        self.similarity = similarity

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            score = self.index_reader.compute_query_document_score(text.metadata['docid'], query.text,
                                                                   similarity=self.similarity)
            if np.isnan(score):
                score = 0
            text.score = score
        return texts
