import numpy as np
import math
from pyserini.index import IndexReader
import functools


class IsMethod:
    def __init__(self,method_name):
        self.method_name=method_name
    def calc_qpp_feature(self,query,**ctx):
        return self.method_name==ctx["method"]

class TurnJaccard:
    def __init__(self,english):
        self.english=english

    def calc_qpp_feature(self,query,**ctx):
        # t.is_stop or
        q_wordset = set([t.lemma_ for t in self.english(query) if not ( t.is_punct)])
        #print(ctx["qid"],query,q_wordset)
        raw_wordset= set([t.lemma_ for t in self.english(ctx["turn_text"]) if not (t.is_punct)])
        #print(ctx["qid"], ctx["turn_text"], raw_wordset)
        return len(q_wordset.intersection(raw_wordset)) / len(q_wordset.union(raw_wordset))


class TurnNumber():
    def __init__(self,col):
        self.sep_token="#" if col=="or_quac" else "_"
    def calc_qpp_feature(self, query, **ctx):
        return int(ctx["qid"].split(self.sep_token)[1])

class QueryLen():
    def calc_qpp_feature(self,query,**ctx):
        return 1*len(query.split())

class MaxIDF():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.terms_cache=term_cache

    def calc_qpp_feature(self,query,**ctx):
        return max(_extact_idf(query, self.index_reader,self.terms_cache))

class AvgIDF():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.terms_cache = term_cache
    def calc_qpp_feature(self,query,**ctx):
        idf_vals=_extact_idf(query, self.index_reader,self.terms_cache)
        idf_vals=[x for x in idf_vals if x>0]
        if len(idf_vals)==0:
            return 0
        return np.mean(idf_vals)

def _extact_idf(query,index_reader,terms_idf_cache):

    stats=index_reader.stats()
    res=[]
    for term in query.split():
        analyzed = index_reader.analyze(term)
        if len(analyzed)==0:
            continue
        if analyzed[0] in terms_idf_cache:
            res.append(terms_idf_cache[analyzed[0]])
            continue
        df, cf = index_reader.get_term_counts(analyzed[0],analyzer=None)
        idf=math.log(stats["documents"]/df) if df>0 else 0
        terms_idf_cache[analyzed[0]]=idf
        res.append(idf)
    return res


def _calc_scq(query,index_reader,term_scq_cache={}):
    stats=index_reader.stats()
    res=[]
    for term in query.split():
        analyzed = index_reader.analyze(term)
        if len(analyzed)==0:
            continue
        if analyzed[0] in term_scq_cache:
            res.append(term_scq_cache[analyzed[0]])
            continue
        df, cf = index_reader.get_term_counts(analyzed[0],analyzer=None)
        scq=(1+math.log(cf))*math.log(1+stats["documents"]/df) if df>0 and cf>0 else 0
        term_scq_cache[analyzed[0]]=scq
        res.append(scq)
    return res

class MaxSCQ():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.terms_cache = term_cache

    def calc_qpp_feature(self,query,**ctx):
        return max(_calc_scq(query, self.index_reader,self.terms_cache))

class AvgSCQ():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.terms_cache = term_cache
    def calc_qpp_feature(self,query,**ctx):
        scq_vals=_calc_scq(query, self.index_reader,self.terms_cache)
        scq_vals=[x for x in scq_vals if x>0]
        if len(scq_vals)==0:
            return 0
        return np.mean(scq_vals)

class MaxVar():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.term_cache=term_cache

    def calc_qpp_feature(self,query,**ctx):
        res=max(_calc_var(query,self.index_reader,self.term_cache))
        return res

class AvgVar():
    def __init__(self,index_reader,term_cache={}):
        self.index_reader=index_reader
        self.term_cache=term_cache
    def calc_qpp_feature(self,query,**ctx):
        var_vals = _calc_var(query,self.index_reader,self.term_cache)
        var_vals = [x for x in var_vals if x>0]
        if len(var_vals)==0:
            return 0
        res=np.mean(var_vals)
        return res

def _calc_var(query,index_reader,terms_var_cache):
    res=[]
    stats = index_reader.stats()
    for term in query.split():
        analyzed = index_reader.analyze(term)
        if len(analyzed)==0:
            continue
        if analyzed[0] in terms_var_cache:
            res.append(terms_var_cache[analyzed[0]])
            continue
        df, cf = index_reader.get_term_counts(analyzed[0],analyzer=None)
        tf_idfs=[]
        print(term,analyzed[0],df)
        if df==0:
            res.append(0)
            terms_var_cache[analyzed[0]]=0
            continue
        for posting in index_reader.get_postings_list(analyzed[0],analyzer=None):
            tf_idfs.append(1+math.log(posting.tf)*math.log(stats["documents"]/df))
        print(term,df,len(tf_idfs))
        tf_idf_mean=np.mean(tf_idfs)
        var=math.sqrt(np.mean(([(x-tf_idf_mean)**2 for x in tf_idfs])))
        res.append(var)
        terms_var_cache[analyzed[0]]=var
    return res
