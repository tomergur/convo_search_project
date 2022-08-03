class PrevUtteranceRewriter():
    def __init__(self, k=1,use_sep_token=True):
        self.k = k
        self.sep_token=" [SEP] " if use_sep_token else " "

    def rewrite(self, query, **ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            return self.sep_token.join(ctx['history'][-self.k:]+[ctx['canonical_rsp'][-1],query])
        return  self.sep_token.join(ctx['history'][-self.k:]+[query])
