import pandas as pd
class FileRewriter():
    def __init__(self,queries_rewrites_path):
        df=pd.read_csv(queries_rewrites_path,header=None,delimiter='\t')
        self.rewrites=df.set_index(0)[1].to_dict()

    def rewrite(self,query,**ctx):
        return self.rewrites.get(ctx['qid'],query)
