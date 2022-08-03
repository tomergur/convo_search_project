from chatty_goose.cqr import Hqe
class HqeRewriter():
    def __init__(self,searcher,use_sep_token=True):
        self.hqe=Hqe(searcher)
        self.use_sep_token=use_sep_token

    def rewrite(self,query,**ctx):
        if len(ctx['history'])==0:
            self.reset_history()
        rewrite=self.hqe.rewrite(query)
        if self.use_sep_token:
            sub_query=rewrite.split(query)[0]
            if len(sub_query)==0:
                return query
            print(rewrite,"sub:",sub_query,"sep",query)
            return query+ " [SEP] " + sub_query
        return rewrite
    def reset_history(self):
        print("reset history")
        self.hqe.reset_history()