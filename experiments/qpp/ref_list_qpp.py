import numpy as np



class CombinedRefListQPP:
    def __init__(self,qpp_predictor,n=10,lambd=0.1,lambd2=0):
        self.qpp_predictor=qpp_predictor
        self.n=n
        self.lambd=lambd
        self.lambd2=lambd2

    def calc_rbo(self,lst, lst2, p=0.95, interpolate=True):
        res = 0
        docs = set()
        #docs=[]
        cutoff = min([len(lst), len(lst2), self.n])
        docs_inter=0
        for i in range(self.n):
            docs.add(lst[i][0])
            #docs.append(lst[i][0])
            cur_lst2_doc=lst2[i][0]
            if cur_lst2_doc in docs:
                docs_inter+=1
            overlap = float(docs_inter) / (i + 1.)
            weight = (1 - p) * (p ** (i))
            res = res + overlap * weight
        if interpolate:
            res = res + overlap * (p ** cutoff)
        return res

    def calc_qpp_feature(self,query,**ctx):
        orig_predictor_val=self.qpp_predictor.calc_qpp_feature(query,**ctx)
        if self.lambd == 0:
            return orig_predictor_val
        rewrites_ref_qpp_vals=[]
        rewrite_refs=ctx["ref_rewrites"]
        for i,ref_tup in enumerate(rewrite_refs):
            ref_query, ref_ctx=ref_tup
            ref_q_qpp_val=self.qpp_predictor.calc_qpp_feature(ref_query,**ref_ctx)
            ref_q_similarity=self.calc_rbo(ctx["res_list"],ref_ctx["res_list"])
            rewrites_ref_qpp_vals.append(ref_q_similarity*ref_q_qpp_val)
        rewrites_avg_scores=np.mean(rewrites_ref_qpp_vals)
        history_refs=ctx["history"]
        if len(history_refs)==0 or self.lambd2==0:
            return (1-self.lambd)*orig_predictor_val+self.lambd*rewrites_avg_scores
        history_ref_qpp_vals=[]
        for i,ref_tup in enumerate(history_refs):
            ref_query, ref_ctx=ref_tup
            ref_q_qpp_val=self.qpp_predictor.calc_qpp_feature(ref_query,**ref_ctx)
            ref_q_similarity=self.calc_rbo(ctx["res_list"],ref_ctx["res_list"])
            history_ref_qpp_vals.append(ref_q_similarity*ref_q_qpp_val)
        histoy_avg_score=np.mean(history_ref_qpp_vals)
        ref_queries_scores=self.lambd2*histoy_avg_score+(1-self.lambd2)*rewrites_avg_scores
        return (1-self.lambd)*orig_predictor_val+self.lambd*ref_queries_scores











class RefListQPP:
    def __init__(self,qpp_predictor,ref_ctx_field_name="history",decay=None,n=10,lambd=0.1,ref_limit=None,method_type=None):
        self.qpp_predictor=qpp_predictor
        self.ref_ctx_field_name=ref_ctx_field_name
        self.n=n
        self.lambd=lambd
        self.decay=decay
        self.ref_limit=ref_limit
        assert(ref_limit is None or ref_ctx_field_name=="history")
        self.method_type=method_type

    def calc_rbo(self,lst, lst2, p=0.95, interpolate=True):
        res = 0
        docs = set()
        #docs=[]
        cutoff = min([len(lst), len(lst2), self.n])
        docs_inter=0
        for i in range(self.n):
            docs.add(lst[i][0])
            #docs.append(lst[i][0])
            cur_lst2_doc=lst2[i][0]
            if cur_lst2_doc in docs:
                docs_inter+=1
            overlap = float(docs_inter) / (i + 1.)
            weight = (1 - p) * (p ** (i))
            res = res + overlap * weight
        if interpolate:
            res = res + overlap * (p ** cutoff)
        return res

    def _calc_rbo(self,lst, lst2, p=0.95, interpolate=True):
        res = 0
        docs, docs2 = set(), set()
        overlap = 0
        cutoff = min([len(lst), len(lst2), self.n])
        for i in range(self.n):
            docs.add(lst[i][0])
            docs2.add(lst2[i][0])
            docs_inter = docs.intersection(docs2)
            overlap = float(len(docs_inter)) / (i + 1.)
            weight = (1 - p) * (p ** (i))
            res = res + overlap * weight
        if interpolate:
            res = res + overlap * (p ** cutoff)
        return res


    def calc_qpp_feature(self,query,**ctx):
        orig_predictor_val=self.qpp_predictor.calc_qpp_feature(query,**ctx)
        refs=ctx[self.ref_ctx_field_name]
        if self.ref_limit is not None:
            refs=refs[-self.ref_limit:]
            #print(ctx['qid'],[q[1]['qid'] for q in refs])
        if self.method_type is not None:
            refs=[r for r in refs if r[1]["method"] in self.method_type]
            #print([r[1]["method"] for r in refs])
        if len(refs) == 0 or self.lambd==0:
            return orig_predictor_val
        ref_qpp_vals=[]
        if self.decay:
            ref_q_weights = [np.exp(-self.decay * (len(refs)-i)) for i in range(len(refs))]
        else:
            ref_q_weights=None
        for i,ref_tup in enumerate(refs):
            ref_query, ref_ctx=ref_tup
            ref_q_qpp_val=self.qpp_predictor.calc_qpp_feature(ref_query,**ref_ctx)
            ref_q_similarity=self.calc_rbo(ctx["res_list"],ref_ctx["res_list"])
            '''
            if self.decay is not None:
                decay_weight=ref_q_weights[i]
                ref_q_similarity=ref_q_similarity*decay_weight
            '''
            ref_qpp_vals.append(ref_q_similarity*ref_q_qpp_val)
        return self.lambd*np.average(ref_qpp_vals,weights=ref_q_weights)+(1-self.lambd)*orig_predictor_val

class HistQPP:
    def __init__(self,qpp_predictor,ref_ctx_field_name="history",sumnormalize=False,decay=1,lambd=0.1):
        self.qpp_predictor=qpp_predictor
        self.ref_ctx_field_name=ref_ctx_field_name
        self.sumnormalize=sumnormalize
        self.decay_factor=decay
        self.lambd=lambd

    def calc_qpp_feature(self,query,**ctx):
        orig_predictor_val=self.qpp_predictor.calc_qpp_feature(query,**ctx)
        refs=ctx[self.ref_ctx_field_name]
        if len(refs) == 0 or self.lambd==0:
            return orig_predictor_val
        ref_qpp_vals=[]
        ref_q_weights=[np.exp(-self.decay_factor*i) for i in range (1,len(refs)+1)]
        if self.sumnormalize:
            ref_q_weights_sum=np.sum(ref_q_weights)
            ref_q_weights=[x/ref_q_weights_sum for x in ref_q_weights]

        for ref_q_weight,ref_tup in zip(ref_q_weights,refs[::-1]):
            ref_query, ref_ctx = ref_tup
            ref_q_qpp_val=self.qpp_predictor.calc_qpp_feature(ref_query,**ref_ctx)
            #ref_q_weight=np.exp(-self.decay_factor*i)
            ref_qpp_vals.append(ref_q_weight*ref_q_qpp_val)
        ref_qpp_res=np.sum(ref_qpp_vals) if self.sumnormalize else np.mean(ref_qpp_vals)
        return self.lambd*ref_qpp_res+(1-self.lambd)*orig_predictor_val