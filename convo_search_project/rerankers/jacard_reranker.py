from pygaggle.rerank.base import Reranker, Query, Text
from copy import deepcopy
from typing import List
import spacy
import numpy as np
class JaacardReranker(Reranker):
    def __init__(self,agg_func):
        self.agg_func=None
        if agg_func is not None:
            self.agg_func=max if agg_func=="max" else np.mean
        self.english=spacy.load("en_core_web_sm",disable=['ner','tagger','parser'])

    def calc_jaccard_coef(self,word_set,word_set2):
        return len(word_set.intersection(word_set2))/len(word_set.union(word_set2))

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        q_wordset = set([t.lemma_ for t in self.english(query.text) if not (t.is_stop or t.is_punct)])
        for text in texts:
            if self.agg_func is not None:
                segments=text.text.split("[SEP]")
                other_wordsets=[set([t.lemma_ for t in self.english(seg) if not (t.is_stop or t.is_punct)]) for seg in segments]
                scores=[self.calc_jaccard_coef(q_wordset,w_set) for w_set in other_wordsets]
                score=self.agg_func(scores)
            else:
                seg=text.text.replace("[SEP]", "")
                other_wordset=set([t.lemma_ for t in self.english(seg) if not (t.is_stop or t.is_punct)])
                score = self.calc_jaccard_coef(q_wordset,other_wordset)
            text.score = score
        return texts

