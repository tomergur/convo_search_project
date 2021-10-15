class PrevAndFirstUtteranceRewriter():
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            if len(ctx['history']) == 1:
                return " ".join([ctx['history'][0], ctx['canonical_rsp'][-1], query])
            return " ".join([ctx['history'][0],ctx['history'][-1],ctx['canonical_rsp'][-1],query])
        if len(ctx['history']) == 1:
            return " ".join([ctx['history'][0], query])
        return " ".join([ctx['history'][0],ctx['history'][-1],query])