from chatty_goose.cqr import Hqe
class HqeRewriter():
    def __init__(self,searcher):
        self.hqe=Hqe(searcher)

    def rewrite(self,query,**ctx):
        return self.hqe.rewrite(query)
    def reset_history(self):
        self.hqe.reset_history()