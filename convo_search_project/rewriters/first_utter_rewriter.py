class FirstUtteranceRewriter():

    def __init__(self,use_sep_token):
        self.sep_token=" [SEP] " if use_sep_token else " "

    def rewrite(self,query,**ctx):
        if len(ctx['history']) == 0:
            return query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            return self.sep_token.join([ctx['history'][0],ctx['canonical_rsp'][-1],query])
        return ctx['history'][0]+self.sep_token+query