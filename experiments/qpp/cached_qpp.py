import json

class KeyValueQPP:
    def __init__(self,cache):
        self.cache=cache
    def calc_qpp_feature(self,query,**ctx):
        method_name=ctx["method"]
        qid=ctx["qid"]
        return self.cache[method_name][qid]


class CachedQPP:
    def __init__(self,predictor,feature_path,**params):
        self.predictor = predictor
        with open(feature_path) as f:
            res_dicts=json.load(f)
        if len(params)==0:
            self.cached_feature=res_dicts
            return
        params_set=set(params.items())
        cached_features={}
        for method,method_feature_val in res_dicts.items():
            for cached_param,feature_cache in method_feature_val:
                cached_params_set=set([tuple(x) for x in cached_param])
                #print("cached params set:",cached_params_set)
                #print(cached_params_set,params_set)
                if cached_params_set.issubset(params_set):
                    #print("found subset",cached_param,params_set)
                    cached_features[method]=feature_cache
        self.cached_feature=cached_features

    def calc_qpp_feature(self,query,**ctx):
        method_name=ctx["method"]
        qid=ctx["qid"]
        if qid not in self.cached_feature[method_name]:
            res=self.predictor.calc_qpp_feature(query,**ctx)
            self.cached_feature[method_name][qid]=res
            return res
        return self.cached_feature[method_name][qid]
