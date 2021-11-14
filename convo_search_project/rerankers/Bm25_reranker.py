from pygaggle.rerank.base import Reranker, Query, Text
from copy import deepcopy
from typing import List
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
