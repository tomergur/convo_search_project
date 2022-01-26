class AllHistoryRewriter():
    def __init__(self,turn_sep=" "):
        self.turn_sep=turn_sep
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            return self.turn_sep.join(ctx['history']+[ctx['canonical_rsp'][-1]]+[query])
        return self.turn_sep.join(ctx['history']+[query])
