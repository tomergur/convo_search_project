class FirstUtteranceRewriter():
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            return " ".join([ctx['history'][0],ctx['canonical_rsp'][-1],query])
        return ctx['history'][0]+" "+query