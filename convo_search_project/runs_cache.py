import os
import json
from types import SimpleNamespace
class RunsCache:
    def _load_intial_lists(self):
        res = {}
        with open(self.cache_path) as f:
            intial_lists = json.load(f)
        for qid, q_res in intial_lists.items():
            # q_simple= [IndexHit(d["docid"],d["score"],d["content"]) for d in q_res]
            q_simple = [SimpleNamespace(docid=d["docid"], score=d["score"], raw=d["content"]) for d in q_res]
            res[qid] = q_simple
        return res

    def __init__(self,cache_path):
        self.cache_path=cache_path
        self.lazy_loading=os.path.isdir(cache_path)
        if not self.lazy_loading:
            self.intial_lists=self._load_intial_lists()

    def __getitem__(self,qid):
        if self.lazy_loading:
            with open("{}/{}.json".format(self.cache_path,qid)) as f:
                q_res= json.load(f)
                q_simple = [SimpleNamespace(docid=d["docid"], score=d["score"], raw=d["content"]) for d in q_res]
                return q_simple
        else:
            return self.intial_lists[qid]