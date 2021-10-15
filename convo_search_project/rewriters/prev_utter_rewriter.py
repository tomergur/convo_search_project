class PrevUtteranceRewriter():
    def __init__(self, k=1):
        self.k = k

    def rewrite(self, query, **ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            return " ".join(ctx['history'][-self.k:]+[ctx['canonical_rsp'][-1],query])
        return  " ".join(ctx['history'][-self.k:]+[query])
