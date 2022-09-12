from .collection_lm import CollectionLM
from .lm_utils import create_doc_dir_smooth_lm, create_doc_lm, create_q_lm
import math
import numpy as np
from scipy.special import softmax
from .rm import calc_rm, score_to_rm
import time


class Clairty:
    def __init__(self, index_reader, doc_lm_cache, dir_doc_cache, collection_lm=None, k=10):
        self.index_reader = index_reader
        self.collection_lm = CollectionLM(index_reader, True) if collection_lm is None else collection_lm
        self.k = k
        self.doc_lm_cache = doc_lm_cache
        self.dir_doc_cache = dir_doc_cache

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        doc_ids = [x[0] for x in res_list[:self.k]]
        query_terms = []
        for term in query.split():
            analyzed = self.index_reader.analyze(term)
            if len(analyzed) == 0:
                continue
            if self.collection_lm[analyzed[0]] == 0:
                continue
            query_terms.append(analyzed[0])
        # doc_time=time.time()
        # docs_lm = {doc_id: create_doc_lm(doc_id, self.index_reader) for doc_id in doc_ids}
        docs_dir_lms = {}
        for doc_id in doc_ids:
            if doc_id in self.dir_doc_cache:
                docs_dir_lms[doc_id] = self.dir_doc_cache[doc_id]
            else:
                doc = create_doc_dir_smooth_lm(doc_id, self.index_reader, self.collection_lm)
                docs_dir_lms[doc_id] = doc
                self.dir_doc_cache[doc_id] = doc

        docs_lm = {}
        for doc_id in doc_ids:
            if doc_id in self.doc_lm_cache:
                docs_lm[doc_id] = self.doc_lm_cache[doc_id]
            else:
                doc = create_doc_lm(doc_id, self.index_reader)
                docs_lm[doc_id] = doc
                self.doc_lm_cache[doc_id] = doc
        # print("load doc time:",time.time()-doc_time)
        rm_model = calc_rm(docs_dir_lms, docs_lm, query_terms)
        # rm_model=get_rm(query_terms,doc_ids,self.index_reader,self.collection_lm)
        # print({w:(rm_model[w],self.collection_lm[w],self.index_reader.get_term_counts(w, analyzer=None)) for w in rm_model.get_all_terms() if self.collection_lm[w]==0})

        clarity_score = math.fsum(
            [rm_model[w] * math.log(rm_model[w] / self.collection_lm[w]) for w in rm_model.get_all_terms()])
        return clarity_score


class ClairtyNorm:
    def __init__(self, index_reader, doc_lm_cache, collection_lm=None, k=10):
        self.index_reader = index_reader
        self.collection_lm = CollectionLM(index_reader, True) if collection_lm is None else collection_lm
        self.k = k
        self.doc_lm_cache = doc_lm_cache

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        doc_ids = [x[0] for x in res_list[:self.k]]
        score_dist = [x[1] for x in res_list[:self.k]]
        max_score = max(score_dist)
        min_score = min(score_dist)
        normalized_scores = [(x - min_score) / (max_score - min_score) for x in score_dist]
        #normalized_scores = softmax(score_dist)
        doc_scores = {x: v for x, v in zip(doc_ids, normalized_scores)}
        query_terms = []
        for term in query.split():
            analyzed = self.index_reader.analyze(term)
            if len(analyzed) == 0:
                continue
            if self.collection_lm[analyzed[0]] == 0:
                continue
            query_terms.append(analyzed[0])

        docs_lm = {}
        for doc_id in doc_ids:
            if doc_id in self.doc_lm_cache:
                docs_lm[doc_id] = self.doc_lm_cache[doc_id]
            else:
                doc = create_doc_lm(doc_id, self.index_reader)
                docs_lm[doc_id] = doc
                self.doc_lm_cache[doc_id] = doc
        rm_model = score_to_rm(doc_scores, docs_lm)
        #print(ctx['qid'])
        #print(rm_model.terms_dict)
        clarity_score = math.fsum(
            [rm_model[w] * math.log(rm_model[w] / self.collection_lm[w]) for w in rm_model.get_all_terms()])
        return clarity_score


class NQCNorm:
    def __init__(self, k=10):
        self.k = k

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        score_dist = [x[1] for x in res_list[:self.k]]
        max_score = max(score_dist)
        min_score = min(score_dist)
        normalized_scores = [(x - min_score) / (max_score - min_score) for x in score_dist]
        #normalized_scores = softmax(score_dist)
        # print(score_dist)
        score_var = np.std(normalized_scores)
        return score_var


class WIGNorm:
    def __init__(self, k=10):
        self.k = k

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        score_dist = [x[1] for x in res_list[:self.k]]
        max_score = max(score_dist)
        min_score = min(score_dist)
        normalized_scores = [(x - min_score) / (max_score - min_score) for x in score_dist]
        #normalized_scores=softmax(score_dist)
        # print(score_dist)
        score_mean = np.mean(normalized_scores)
        return score_mean


class WIG:
    def __init__(self, index_reader, doc_cache, collection_lm=None, k=10):
        self.index_reader = index_reader
        self.collection_lm = CollectionLM(index_reader, True) if collection_lm is None else collection_lm
        self.doc_cache = doc_cache
        self.k = k

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        doc_ids = [x[0] for x in res_list[:self.k]]
        docs = []
        for doc_id in doc_ids:
            if doc_id in self.doc_cache:
                docs.append(self.doc_cache[doc_id])
            else:
                doc = create_doc_dir_smooth_lm(doc_id, self.index_reader, self.collection_lm)
                docs.append(doc)
                self.doc_cache[doc_id] = doc

        # docs=[create_doc_dir_smooth_lm(doc_id,self.index_reader,self.collection_lm) for doc_id in doc_ids]
        query_terms = []
        for term in query.split():
            analyzed = self.index_reader.analyze(term)
            if len(analyzed) == 0:
                continue
            if self.collection_lm[analyzed[0]] == 0:
                continue
            query_terms.append(analyzed[0])
        wig_score = 0
        collection_penalty = math.fsum([math.log(self.collection_lm[w]) for w in query_terms])
        for doc in docs:
            wig_doc_score = math.fsum([math.log(doc[w]) for w in query_terms]) - collection_penalty
            wig_score += wig_doc_score
        return wig_score / (self.k * math.sqrt(len(query_terms)))


class NQC:
    def __init__(self, index_reader, doc_cache, collection_lm=None, k=10):
        self.index_reader = index_reader
        self.collection_lm = CollectionLM(index_reader, True) if collection_lm is None else collection_lm
        self.doc_cache = doc_cache
        self.k = k

    def calc_qpp_feature(self, query, **ctx):
        res_list = ctx["res_list"]
        doc_ids = [x[0] for x in res_list[:self.k]]
        docs = []
        for doc_id in doc_ids:
            if doc_id in self.doc_cache:
                docs.append(self.doc_cache[doc_id])
            else:
                doc = create_doc_dir_smooth_lm(doc_id, self.index_reader, self.collection_lm)
                docs.append(doc)
                self.doc_cache[doc_id] = doc

        query_terms = []
        for term in query.split():
            analyzed = self.index_reader.analyze(term)
            if len(analyzed) == 0:
                continue
            if self.collection_lm[analyzed[0]] == 0:
                continue
            query_terms.append(analyzed[0])
        collection_penalty = math.fsum([math.log(self.collection_lm[w]) for w in query_terms])
        qld = []
        for doc in docs:
            doc_score = math.fsum([math.log(doc[w]) for w in query_terms])
            qld.append(doc_score)
        return np.std(qld) / collection_penalty
