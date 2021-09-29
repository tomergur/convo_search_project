import pandas as pd
import json


class FileRewriter():
    def _load_tsv_rewrites(self, queries_rewrites_path):
        df = pd.read_csv(queries_rewrites_path, header=None, delimiter='\t')
        return df.set_index(0)[1].to_dict()

    def _load_json_rewrites(self, queries_rewrites_path):
        with open(queries_rewrites_path) as f:
            rewrites = json.load(f)
            rewrites = {qid: v["first_stage_rewrites"] for qid, v in rewrites.items()}
        return rewrites

    def __init__(self, queries_rewrites_path):
        if queries_rewrites_path.endswith("tsv"):
            self.rewrites = self._load_tsv_rewrites(queries_rewrites_path)
        else:
            self.rewrites = self._load_json_rewrites(queries_rewrites_path)

    def rewrite(self, query, **ctx):
        return self.rewrites.get(ctx['qid'], query)
