class AllHistoryRewriter():
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        return " ".join([query]+ctx['history'])
