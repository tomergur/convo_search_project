
class CollectionLM:
    def __init__(self,index_reader,cache_res=False):
        self.index_reader=index_reader
        stats=index_reader.stats()
        self.total_terms=stats["total_terms"]
        self.terms_cache={}
        self.cache_res=cache_res

    def __getitem__(self, term):
        if self.cache_res and term in self.terms_cache:
            return self.terms_cache[term]/float(self.total_terms)

        df, cf = self.index_reader.get_term_counts(term, analyzer=None)

        if self.cache_res:
            self.terms_cache[term]=cf
            #print("cache",len(self.terms_cache))
        return float(cf) / float(self.total_terms)
