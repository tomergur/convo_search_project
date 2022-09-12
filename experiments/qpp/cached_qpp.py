import json
class CachedQPP:
    def __init__(self,feature_path,**params):
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
        return self.cached_feature[method_name][qid]
