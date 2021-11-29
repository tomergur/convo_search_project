from chatty_goose.cqr import Hqe
class HqeRewriter():
    def __init__(self,searcher):
        self.hqe=Hqe(searcher)
    def rewrite(self,query,**ctx):
        if len(ctx['history'])==0:
            self.reset_history()
        return self.hqe.rewrite(query)
    def reset_history(self):
        print("reset history")
        self.hqe.reset_history()