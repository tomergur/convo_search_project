import math
import heapq
import scipy as scp


class UnigramLM:
    def __init__(self, terms_dict,sumnormalize=False):
        self.terms_dict = self._sumnormalize(terms_dict) if sumnormalize else terms_dict

    def __getitem__(self, term):
        return self.terms_dict.get(term, .0)

    def get_all_terms(self):
        return self.terms_dict.keys()

    def calc_ce_score(self, lm):
        ce_vals = [self[t] * math.log(lm[t]) for t in self.get_all_terms()]
        return math.fsum(ce_vals)
    def calc_kld(self,lm):
        kld_vals=[-1*self[t] * math.log(self[t]/lm[t]) for t in self.get_all_terms()]
        return math.fsum(kld_vals)

    def calc_query_likelihood(self,query_terms):
        res=1
        for term in query_terms:
            res*=self.terms_dict[term]
        return res


    def clip_model(self, num):
        if num is None:
            return UnigramLM(self._sumnormalize(self.terms_dict))
        # best_terms=heapq.nlargest(num,self.terms_dict.items(),key=lambda t:t[1])
        best_terms = heapq.nlargest(num, self.terms_dict.items(), key=lambda t: (t[1], t[0]))
        clipped_dict = dict(best_terms)
        return UnigramLM(self._sumnormalize(clipped_dict))

    def clip_model_by_terms(self, terms, use_weights=True):
        best_terms = {t: self.terms_dict[t] if use_weights else 1 for t in terms}
        return UnigramLM(self._sumnormalize(best_terms))

    def get_entropy(self):
        return scp.stats.entropy(list(self.terms_dict.values()))

    def _sumnormalize(self, term_dict):
        term_sum = math.fsum(term_dict.values())
        return {t: prob / term_sum for (t, prob) in term_dict.items()}

    @staticmethod
    def interpolate(lm1, lm2, lm1_weight):
        w1 = lm1_weight
        w2 = 1 - w1
        lm1_terms = set(lm1.get_all_terms())
        lm2_terms = set(lm2.get_all_terms())
        terms = lm1_terms | lm2_terms
        new_lm = {t: w1 * lm1[t] + w2 * lm2[t] for t in terms}
        return UnigramLM(new_lm)