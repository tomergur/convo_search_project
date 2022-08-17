class AllHistoryRewriter():
    def __init__(self,turn_sep=" ",rsp_context_type="type_a"):
        self.turn_sep=turn_sep
        self.rsp_context_type=rsp_context_type
    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            if self.rsp_context_type=="type_b":
                return self.turn_sep.join(ctx['history']+[ctx['canonical_rsp'][-1]]+[query])
            merged_hist=[]
            for i in range(len(ctx['history'])):
                merged_hist.append(ctx['history'][i])
                canonical_turn_rsp=ctx['canonical_rsp'][i]
                if canonical_turn_rsp is not None:
                    merged_hist.append(canonical_turn_rsp)
            return self.turn_sep.join(merged_hist+[query])
        return self.turn_sep.join(ctx['history']+[query])
