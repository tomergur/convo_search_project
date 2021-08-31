class SeparateUtterancesRewriter():
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        return [query]+ctx['history']