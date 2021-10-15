class AllHistoryRewriter():
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx:
            return " ".join(ctx['history']+[ctx['canonical_rsp'][-1]]+[query])
        return " ".join(ctx['history']+[query])
