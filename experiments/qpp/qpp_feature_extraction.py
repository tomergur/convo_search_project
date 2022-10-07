
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
import numpy as np
import scipy.stats
import functools
import spacy
from .pre_ret_qpp import QueryLen,TurnNumber,MaxIDF,AvgIDF,MaxSCQ,AvgSCQ,MaxVar,AvgVar,IsMethod,TurnJaccard
from .post_ret_qpp import WIG,Clairty,NQC,NQCNorm,WIGNorm,ClairtyNorm
from .ref_list_qpp import RefListQPP,HistQPP
from .cached_qpp import CachedQPP
from .bert_qpp_infer import BertQPP
from .collection_lm import CollectionLM


class QPPFeatureFactory:
    def __init__(self,col='cast19',cached_features_path=None):
        self.cached_features_path=cached_features_path
        if 'cast' in col:
            self.index_reader = IndexReader.from_prebuilt_index('cast2019')
            self.searcher=SimpleSearcher.from_prebuilt_index('cast2019')
        elif col=="or_quac":
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/or_quac")
            self.searcher=SimpleSearcher("/v/tomergur/convo/indexes/or_quac")
        elif col=="reddit":
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/reddit")
            self.searcher=SimpleSearcher("/v/tomergur/convo/indexes/reddit")
        else:
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/topiocqa")
            self.searcher = SimpleSearcher("/v/tomergur/convo/indexes/topiocqa")
        self.collection_lm=CollectionLM(self.index_reader,True)
        self.english=spacy.load("en_core_web_sm",disable=['ner','tagger','parser'])
        qpp_dict = {"q_len":lambda: QueryLen(),"turn_number":lambda:TurnNumber(col)}
        index_reader = self.index_reader
        self.terms_idf_cache={}
        qpp_dict["is_t5"]=lambda: IsMethod("t5")
        qpp_dict["is_all"] = lambda: IsMethod("all")
        qpp_dict["is_hqe"] = lambda: IsMethod("hqe")
        qpp_dict["is_quretec"] = lambda: IsMethod("quretec")
        qpp_dict["turn_jaccard"] = lambda: TurnJaccard(self.english)
        qpp_dict["max_idf"] = lambda:MaxIDF(index_reader,self.terms_idf_cache)
        qpp_dict["avg_idf"] = lambda:AvgIDF(index_reader,self.terms_idf_cache)
        self.terms_scq_cache={}
        qpp_dict["max_scq"] = lambda:MaxSCQ(index_reader,self.terms_scq_cache)
        qpp_dict["avg_scq"] = lambda:AvgSCQ(index_reader,self.terms_scq_cache)
        self.terms_var_cache={}
        qpp_dict["max_var"] = lambda:MaxVar(index_reader,self.terms_var_cache)
        qpp_dict["avg_var"] = lambda:AvgVar(index_reader,self.terms_var_cache)
        self.dir_doc_cache={}
        self.doc_lm_cache={}
        qpp_dict["bert_qpp"]=lambda: BertQPP(self.searcher,"/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}",col)
        qpp_dict["bert_qpp_or_quac"] = lambda: BertQPP(self.searcher, "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}", "or_quac")
        qpp_dict["bert_qpp_topiocqa"] = lambda: BertQPP(self.searcher, "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}", "topiocqa")
        qpp_dict["bert_qpp_hist"] = lambda: BertQPP(self.searcher, "/v/tomergur/convo/qpp_models/bert_qpp_hist_rerank/{}_{}", col,True)
        qpp_dict["bert_qpp_hist_or_quac"] = lambda: BertQPP(self.searcher, "/v/tomergur/convo/qpp_models/bert_qpp_hist_rerank/{}_{}", "or_quac",True)
        qpp_dict["bert_qpp_hist_topiocqa"] = lambda: BertQPP(self.searcher, "/v/tomergur/convo/qpp_models/bert_qpp_hist_rerank/{}_{}", "topiocqa",True)
        qpp_dict["bert_qpp_prev"] = lambda: BertQPP(self.searcher,"/v/tomergur/convo/qpp_models/bert_qpp_prev_rerank/{}_{}", col,append_prev_turns=True)
        qpp_dict["WIG"]=lambda hp_config:(WIG(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k']))
        qpp_dict["clarity"] = lambda hp_config: Clairty(self.index_reader,self.doc_lm_cache, self.dir_doc_cache,self.collection_lm, hp_config['k'])
        qpp_dict["NQC"] = lambda hp_config: NQC(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k'])
        qpp_dict["NQC_norm"] = lambda hp_config: NQCNorm(k=hp_config['k'])
        qpp_dict["WIG_norm"] = lambda hp_config: WIGNorm(k=hp_config['k'])
        qpp_dict["clarity_norm"] = lambda hp_config: ClairtyNorm(self.index_reader,self.doc_lm_cache,self.collection_lm, hp_config['k'])
        '''
        qpp_dict["ref_rewrites_max_idf"] = lambda hp_config: RefListQPP(MaxIDF(self.index_reader,self.terms_idf_cache),ref_ctx_field_name="ref_rewrites", **hp_config)
        qpp_dict["ref_rewrites_avg_idf"] = lambda hp_config: RefListQPP(AvgIDF(self.index_reader,self.terms_idf_cache),ref_ctx_field_name="ref_rewrites", **hp_config)
        qpp_dict["ref_rewrites_max_scq"] = lambda hp_config: RefListQPP(MaxSCQ(self.index_reader, self.terms_scq_cache),                                                    ref_ctx_field_name="ref_rewrites", **hp_config)
        qpp_dict["ref_rewrites_avg_scq"] = lambda hp_config: RefListQPP(AvgSCQ(self.index_reader, self.terms_scq_cache),
                                                                        ref_ctx_field_name="ref_rewrites", **hp_config)
        qpp_dict["ref_rewrites_max_var"]=lambda hp_config: RefListQPP(MaxVar(self.index_reader,self.terms_var_cache),ref_ctx_field_name="ref_rewrites",**hp_config)
        qpp_dict["ref_rewrites_avg_var"]=lambda hp_config: RefListQPP(AvgVar(self.index_reader,self.terms_var_cache),ref_ctx_field_name="ref_rewrites",**hp_config)
        qpp_dict["ref_rewrites_WIG"]=lambda hp_config: RefListQPP(WIG(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k']),ref_ctx_field_name="ref_rewrites",n=hp_config["n"],lambd=hp_config["lambd"])
        qpp_dict["ref_rewrites_clarity"]=lambda hp_config: RefListQPP(Clairty(self.index_reader,self.doc_lm_cache, self.dir_doc_cache,self.collection_lm, hp_config['k']),ref_ctx_field_name="ref_rewrites",n=hp_config["n"],lambd=hp_config["lambd"])
        qpp_dict["ref_rewrites_NQC"]=lambda hp_config: RefListQPP(NQC(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k']),ref_ctx_field_name="ref_rewrites",n=hp_config["n"],lambd=hp_config["lambd"])
       
        qpp_dict["ref_hist_max_idf"]=lambda hp_config: RefListQPP(MaxIDF(self.index_reader,self.terms_idf_cache),**hp_config)
        qpp_dict["ref_hist_avg_idf"]=lambda hp_config: RefListQPP(AvgIDF(self.index_reader,self.terms_idf_cache),**hp_config)
        qpp_dict["ref_hist_max_scq"]=lambda hp_config: RefListQPP(MaxSCQ(self.index_reader),**hp_config)
        qpp_dict["ref_hist_avg_scq"]=lambda hp_config: RefListQPP(AvgSCQ(self.index_reader),**hp_config)
        qpp_dict["ref_hist_max_var"]=lambda hp_config: RefListQPP(MaxVar(self.index_reader,self.terms_var_cache),**hp_config)
        qpp_dict["ref_hist_avg_var"]=lambda hp_config: RefListQPP(AvgVar(self.index_reader,self.terms_var_cache),**hp_config)
        qpp_dict["ref_hist_WIG"]=lambda hp_config: RefListQPP(WIG(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k']),n=hp_config["n"],lambd=hp_config["lambd"])
        qpp_dict["ref_hist_clarity"]=lambda hp_config: RefListQPP(Clairty(self.index_reader,self.doc_lm_cache, self.dir_doc_cache,self.collection_lm, hp_config['k']),n=hp_config["n"],lambd=hp_config["lambd"])
        qpp_dict["ref_hist_NQC"]=lambda hp_config: RefListQPP(NQC(self.index_reader, self.dir_doc_cache,self.collection_lm, hp_config['k']),n=hp_config["n"],lambd=hp_config["lambd"])

        #qpp_dict["ref_hist_decay_max_idf"]=lambda hp_config: HistQPP(MaxIDF(self.index_reader,self.terms_idf_cache),**hp_config)
        #qpp_dict["ref_hist_decay_avg_idf"]=lambda hp_config: HistQPP(AvgIDF(self.index_reader,self.terms_idf_cache),**hp_config)
        '''
        self.qpp_dict=qpp_dict

    def _create_hist_factory_func(self,core_feature_name,**params):
        lambd=params["lambd"]
        n=params["n"]
        del params["n"]
        del params["lambd"]
        decay=None
        if "decay" in params:
            decay=params["decay"]
            del params["decay"]
        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return RefListQPP(core_qpp,n=n,lambd=lambd,decay=decay)

    def _create_rewrites_factory_func(self, core_feature_name, **params):
        lambd = params["lambd"]
        n = params["n"]
        del params["n"]
        del params["lambd"]


        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return RefListQPP(core_qpp, n=n, lambd=lambd,ref_ctx_field_name="ref_rewrites")

    def _create_hist_decay_factory_func(self,core_feature_name,sumnormalize,**params):
        lambd = params["lambd"]
        decay = params["decay"]
        del params["decay"]
        del params["lambd"]
        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return HistQPP(core_qpp,decay=decay,lambd=lambd,sumnormalize=sumnormalize)


    def create_qpp_extractor(self,feature_name,**params):
        if len(params)==0:
            if self.cached_features_path is not None:
                cached_feature_file_path = "{}/cache/{}.json".format(self.cached_features_path, feature_name)
                return CachedQPP(cached_feature_file_path)
            return self.qpp_dict[feature_name]()

        if feature_name.startswith("ref_hist_comb_decay"):
            core_feature_name=feature_name.split("ref_hist_comb_decay_")[1]
            return self._create_hist_factory_func(core_feature_name=core_feature_name,**params)

        if feature_name.startswith("ref_hist_decay_norm"):
            core_feature_name=feature_name.split("ref_hist_decay_norm_")[1]
            return self._create_hist_decay_factory_func(core_feature_name=core_feature_name,sumnormalize=True,**params)

        if feature_name.startswith("ref_hist_decay"):
            core_feature_name=feature_name.split("ref_hist_decay_")[1]
            return self._create_hist_decay_factory_func(core_feature_name=core_feature_name,sumnormalize=False,**params)

        if feature_name.startswith("ref_hist"):
            core_feature_name=feature_name.split("ref_hist_")[1]
            return self._create_hist_factory_func(core_feature_name=core_feature_name,**params)

        if feature_name.startswith("ref_rewrites"):
            core_feature_name=feature_name.split("ref_rewrites_")[1]
            return self._create_rewrites_factory_func(core_feature_name=core_feature_name,**params)
        print()
        if self.cached_features_path is not None:
            cached_feature_file_path="{}/cache/{}.json".format(self.cached_features_path,feature_name)
            return CachedQPP(cached_feature_file_path,**params)
        return self.qpp_dict[feature_name](params)



