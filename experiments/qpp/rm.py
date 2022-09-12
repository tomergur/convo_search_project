import math
from .lm import UnigramLM
from .lm_utils import create_doc_lm,create_doc_dir_smooth_lm
import time
def _calc_likelihood(q_terms, doc_lm):
    score = 1.
    for t in q_terms:
        score *= doc_lm[t]
    return score


def get_rm(query_terms, docs_ids,index_reader,c_lm,fb_terms=100):
    start_time=time.time()
    docs_lm = {doc_id:create_doc_lm(doc_id,index_reader) for doc_id in docs_ids}
    docs_dir_lms = {doc_id:create_doc_dir_smooth_lm(doc_id,index_reader,c_lm) for doc_id in docs_ids}
    doc_time = time.time()
    print("finish doc calc:", doc_time - start_time)
    return calc_rm(docs_dir_lms, docs_lm,query_terms,fb_terms)


def calc_rm(docs_dir_lms, docs_lm, query_terms,fb_terms=100):
    doc_scores = {doc_id: _calc_likelihood(query_terms, doc_lm) for doc_id, doc_lm in docs_dir_lms.items()}
    res = score_to_rm(doc_scores, docs_lm, fb_terms)
    return res


def score_to_rm(doc_scores, docs_lm, fb_terms=100):
    scores_sum = math.fsum(doc_scores.values())
    weights = {}
    for doc_id, doc_lm in docs_lm.items():
        current_doc_score=doc_scores[doc_id]
        if current_doc_score==0:
            continue
        for t in doc_lm.get_all_terms():
            weights[t] = weights.get(t, 0.) + (current_doc_score / scores_sum) * doc_lm[t]
    model = UnigramLM(weights)
    res = model.clip_model(fb_terms)
    return res