from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher
import numpy as np
import scipy.stats
import functools
import spacy
from .pre_ret_qpp import QueryLen, TurnNumber, MaxIDF, AvgIDF, MaxSCQ, AvgSCQ, MaxVar, AvgVar, IsMethod, TurnJaccard
from .post_ret_qpp import WIG, Clairty, NQC, NQCNorm, WIGNorm, ClairtyNorm
from .ref_list_qpp import RefListQPP, HistQPP
from .cached_qpp import CachedQPP
from .bert_qpp_infer import BertQPP, GroupwiseBertQPP, SingleTurnBertQPP, BertPLQPP, RewritesBertQPP
from .collection_lm import CollectionLM


class QPPFeatureFactory:
    def __init__(self, col='cast19', cached_features_path=None,searcher_col=None):
        self.cached_features_path = cached_features_path
        self.col = col
        searcher_col=searcher_col if searcher_col is not None else col
        if 'cast' in searcher_col:
            self.index_reader = IndexReader.from_prebuilt_index('cast2019')
            self.searcher = SimpleSearcher.from_prebuilt_index('cast2019')
        elif searcher_col == "or_quac":
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/or_quac")
            self.searcher = SimpleSearcher("/v/tomergur/convo/indexes/or_quac")
        elif searcher_col == "reddit":
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/reddit")
            self.searcher = SimpleSearcher("/v/tomergur/convo/indexes/reddit")
        else:
            self.index_reader = IndexReader("/v/tomergur/convo/indexes/topiocqa")
            self.searcher = SimpleSearcher("/v/tomergur/convo/indexes/topiocqa")
        self.collection_lm = CollectionLM(self.index_reader, True)
        self.english = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser'])
        qpp_dict = {"q_len": lambda: QueryLen(), "turn_number": lambda: TurnNumber(col)}
        index_reader = self.index_reader
        self.terms_idf_cache = {}
        qpp_dict["is_t5"] = lambda: IsMethod("t5")
        qpp_dict["is_all"] = lambda: IsMethod("all")
        qpp_dict["is_hqe"] = lambda: IsMethod("hqe")
        qpp_dict["is_quretec"] = lambda: IsMethod("quretec")
        qpp_dict["turn_jaccard"] = lambda: TurnJaccard(self.english)
        qpp_dict["max_idf"] = lambda: MaxIDF(index_reader, self.terms_idf_cache)
        qpp_dict["avg_idf"] = lambda: AvgIDF(index_reader, self.terms_idf_cache)
        self.terms_scq_cache = {}
        qpp_dict["max_scq"] = lambda: MaxSCQ(index_reader, self.terms_scq_cache)
        qpp_dict["avg_scq"] = lambda: AvgSCQ(index_reader, self.terms_scq_cache)
        self.terms_var_cache = {}
        qpp_dict["max_var"] = lambda: MaxVar(index_reader, self.terms_var_cache)
        qpp_dict["avg_var"] = lambda: AvgVar(index_reader, self.terms_var_cache)
        self.dir_doc_cache = {}
        self.doc_lm_cache = {}
        # qpp_dict["bert_qpp"]=lambda: BertQPP(self.searcher,"/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}",col)
        qpp_dict["bert_pl"] = lambda hp_config: BertPLQPP(self.searcher,
                                                          "/v/tomergur/convo/qpp_models/bert_pl_rerank/{}_{}" +
                                                          hp_config['suffix'], col, 2,10)
        qpp_dict["bert_pl_32"] = lambda hp_config: BertPLQPP(self.searcher,
                                                          "/v/tomergur/convo/qpp_models/bert_pl_rerank_32/{}_{}" +
                                                          hp_config['suffix'], col, 4,32)
        qpp_dict["seq_qpp"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                 "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_seq/{}_{}" +
                                                                 hp_config['suffix'], col, seqQPP=True)
        qpp_dict["seq_qpp_3"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                   "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_seq/{}_{}" +
                                                                   hp_config['suffix'], col, seqQPP=True)
        qpp_dict["many_turns_bert_qpp_cls"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                 "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_cls/{}_{}" +
                                                                                 hp_config['suffix'], col,
                                                                                 output_mode="online_seq")
        qpp_dict["many_turns_bert_qpp_reg"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                 "/v/tomergur/convo/qpp_models_backup/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                 hp_config['suffix'], col,
                                                                                 output_mode="online")
        qpp_dict["many_turns_bert_qpp"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                             "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank/{}_{}" +
                                                                             hp_config['suffix'] + "/text_embed/",
                                                                             "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank/{}_{}" +
                                                                             hp_config['suffix'] + "/group_model/", col,
                                                                             output_mode="online_seq")
        qpp_dict["rewrites_bert_qpp"] = lambda hp_config: RewritesBertQPP(self.searcher,
                                                                          "/v/tomergur/convo/qpp_models/rewrites_bert_qpp_rerank/{}_{}".format(
                                                                              col, hp_config['suffix']))
        qpp_dict["rewrites_bert_qpp_8"] = lambda hp_config: RewritesBertQPP(self.searcher,
                                                                          "/v/tomergur/convo/qpp_models/rewrites_bert_qpp_rerank_8/{}_{}".format(
                                                                              col, hp_config['suffix']),8)
        qpp_dict["many_turns_bert_qpp_tokens_or_quac"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                    "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                    hp_config['suffix'], "or_quac",
                                                                                    output_mode="online")
        qpp_dict["many_turns_bert_qpp_tokens_topiocqa"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                    "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                    hp_config['suffix'], "topiocqa",
                                                                                    output_mode="online")
        qpp_dict["many_turns_bert_qpp_tokens"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                    "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                    hp_config['suffix'], col,
                                                                                    output_mode="online")
        qpp_dict["many_turns_bert_qpp_tokens_l2"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                    "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens_l2/{}_{}" +
                                                                                    hp_config['suffix'], col,
                                                                                    output_mode="online")
        qpp_dict["many_turns_bert_qpp_tokens_skturns"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                            "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens{}/".format(
                                                                                                "_" + str(hp_config[
                                                                                                              'max_seq_length']) if
                                                                                                hp_config[
                                                                                                    'max_seq_length'] is not None else "") + "{}_{}" +
                                                                                            hp_config['suffix'], col,
                                                                                            output_mode="online",
                                                                                            max_seq_length=hp_config[
                                                                                                'max_seq_length'])
        qpp_dict["many_turns_bert_qpp_tokens_1kturns"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                            "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens_1/{}_{}" +
                                                                                            hp_config['suffix'], col,
                                                                                            output_mode="online",
                                                                                            max_seq_length=1)
        qpp_dict["many_turns_bert_qpp_tokens_2kturns"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                            "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens_2/{}_{}" +
                                                                                            hp_config['suffix'], col,
                                                                                            output_mode="online",
                                                                                            max_seq_length=2)
        qpp_dict["many_turns_bert_qpp_tokens_3kturns"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                            "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens_3/{}_{}" +
                                                                                            hp_config['suffix'], col,
                                                                                            output_mode="online",
                                                                                            max_seq_length=3)
        qpp_dict["many_turns_bert_qpp_tokens_3"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                      "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                      hp_config['suffix'], col,
                                                                                      output_mode="online")
        qpp_dict["many_turns_bert_qpp_tokens_init"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                         "/v/tomergur/convo/qpp_models/many_turns_qpp_rerank_tokens/{}_{}" +
                                                                                         hp_config['suffix'], col,
                                                                                         output_mode="online")
        qpp_dict["many_turns_bert_qpp_hist"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                  "/v/tomergur/convo/qpp_models/many_turns_bert_qpp_hist_rerank/{}_{}" +
                                                                                  hp_config['suffix'] + "/text_embed/",
                                                                                  "/v/tomergur/convo/qpp_models/many_turns_bert_qpp_hist_rerank/{}_{}" +
                                                                                  hp_config['suffix'] + "/group_model/",
                                                                                  col, True)
        qpp_dict["many_turns_bert_qpp_prev"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                                  "/v/tomergur/convo/qpp_models/many_turns_bert_qpp_prev_rerank/{}_{}" +
                                                                                  hp_config['suffix'] + "/text_embed/",
                                                                                  "/v/tomergur/convo/qpp_models/many_turns_bert_qpp_prev_rerank/{}_{}" +
                                                                                  hp_config['suffix'] + "/group_model/",
                                                                                  col, append_prev_turns=True)
        qpp_dict["many_docs_bert_qpp"] = lambda hp_config: GroupwiseBertQPP(self.searcher,
                                                                            "/v/tomergur/convo/qpp_models/many_docs_bert_qpp_rerank/{}_{}_" +
                                                                            hp_config['suffix'] + "/text_embed/",
                                                                            "/v/tomergur/convo/qpp_models/many_docs_bert_qpp_rerank/{}_{}_" +
                                                                            hp_config['suffix'] + "/group_model/", col,
                                                                            infer_mode="query",
                                                                            group_agg_func=hp_config['group_agg_func'])
        qpp_dict["bert_qpp"] = lambda hp_config: BertQPP(self.searcher,
                                                         "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}" +
                                                         hp_config['suffix'], col)
        qpp_dict["bert_qpp_or_quac"] = lambda hp_config: BertQPP(self.searcher,
                                                         "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}" +
                                                         hp_config['suffix'], "or_quac")
        qpp_dict["bert_qpp_topiocqa"] = lambda hp_config: BertQPP(self.searcher,
                                                         "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}" +
                                                         hp_config['suffix'], "topiocqa")
        qpp_dict["bert_qpp_3"] = lambda hp_config: BertQPP(self.searcher,
                                                           "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}" +
                                                           hp_config['suffix'], col)
        qpp_dict["st_bert_qpp"] = lambda hp_config: SingleTurnBertQPP(self.searcher,
                                                                      "/v/tomergur/convo/qpp_models/single_turn_bert_qpp_rerank/{}_{}_{}" +
                                                                      hp_config['suffix'], col)
        qpp_dict["bert_qpp_cls"] = lambda hp_config: BertQPP(self.searcher,
                                                             "/v/tomergur/convo/qpp_models_backup/bert_qpp_rerank/{}_{}" +
                                                             hp_config['suffix'], col)
        qpp_dict["bert_qpp_reg"] = lambda hp_config: BertQPP(self.searcher,
                                                             "/v/tomergur/convo/qpp_models/bert_qpp_rerank/{}_{}" +
                                                             hp_config['suffix'], col)
        qpp_dict["bert_qpp_hist"] = lambda hp_config: BertQPP(self.searcher,
                                                              "/v/tomergur/convo/qpp_models/bert_qpp_hist_rerank/{}_{}" +
                                                              hp_config['suffix'], col, True)
        qpp_dict["bert_qpp_prev"] = lambda hp_config: BertQPP(self.searcher,
                                                              "/v/tomergur/convo/qpp_models/bert_qpp_prev_rerank/{}_{}" +
                                                              hp_config['suffix'], col, append_prev_turns=True)

        qpp_dict["WIG"] = lambda hp_config: (
            WIG(self.index_reader, self.dir_doc_cache, self.collection_lm, hp_config['k']))
        qpp_dict["clarity"] = lambda hp_config: Clairty(self.index_reader, self.doc_lm_cache, self.dir_doc_cache,
                                                        self.collection_lm, hp_config['k'])
        qpp_dict["NQC"] = lambda hp_config: NQC(self.index_reader, self.dir_doc_cache, self.collection_lm,
                                                hp_config['k'])
        qpp_dict["NQC_norm"] = lambda hp_config: NQCNorm(k=hp_config['k'])
        qpp_dict["WIG_norm"] = lambda hp_config: WIGNorm(k=hp_config['k'])
        qpp_dict["clarity_norm"] = lambda hp_config: ClairtyNorm(self.index_reader, self.doc_lm_cache,
                                                                 self.collection_lm, hp_config['k'])
        self.qpp_dict = qpp_dict

    def _create_hist_factory_func(self, core_feature_name, **params):
        lambd = params["lambd"]
        n = params["n"]
        ref_limit = None
        if "ref_limit" in params:
            ref_limit = params["ref_limit"]
            del params["ref_limit"]
        del params["n"]
        del params["lambd"]
        decay = None
        if "decay" in params:
            decay = params["decay"]
            del params["decay"]
        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return RefListQPP(core_qpp, n=n, lambd=lambd, decay=decay, ref_limit=ref_limit)

    def _create_rewrites_factory_func(self, core_feature_name, **params):
        lambd = params["lambd"]
        n = params["n"]
        del params["n"]
        del params["lambd"]
        method_type = None
        if "method_type" in params:
            method_type = params["method_type"]
            del params["method_type"]
        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return RefListQPP(core_qpp, n=n, lambd=lambd, ref_ctx_field_name="ref_rewrites", method_type=method_type)

    def _create_hist_decay_factory_func(self, core_feature_name, sumnormalize, **params):
        lambd = params["lambd"]
        decay = params["decay"]
        del params["decay"]
        del params["lambd"]
        core_qpp = self.create_qpp_extractor(core_feature_name, **params)
        return HistQPP(core_qpp, decay=decay, lambd=lambd, sumnormalize=sumnormalize)

    def create_qpp_extractor(self, feature_name, **params):
        if feature_name.startswith("ref_hist_comb_decay"):
            core_feature_name = feature_name.split("ref_hist_comb_decay_")[1]
            return self._create_hist_factory_func(core_feature_name=core_feature_name, **params)

        if feature_name.startswith("ref_hist_decay_norm"):
            core_feature_name = feature_name.split("ref_hist_decay_norm_")[1]
            return self._create_hist_decay_factory_func(core_feature_name=core_feature_name, sumnormalize=True,
                                                        **params)

        if feature_name.startswith("ref_hist_decay"):
            core_feature_name = feature_name.split("ref_hist_decay_")[1]
            return self._create_hist_decay_factory_func(core_feature_name=core_feature_name, sumnormalize=False,
                                                        **params)

        if feature_name.startswith("ref_hist"):
            core_feature_name = feature_name.split("ref_hist_")[1]
            if core_feature_name.endswith("kturns"):
                core_feature_name = core_feature_name[:-8]
            return self._create_hist_factory_func(core_feature_name=core_feature_name, **params)

        if feature_name.startswith("ref_rewrites"):
            core_feature_name = feature_name.split("ref_rewrites_")[1]
            if core_feature_name.endswith("methods"):
                feature_name_arr = core_feature_name.split("_")
                method_name = feature_name_arr[-2]
                core_feature_name = "_".join(feature_name_arr[:-2])
                print("method_name", core_feature_name)
            return self._create_rewrites_factory_func(core_feature_name=core_feature_name, **params)

        if len(params) == 0:
            if self.cached_features_path is not None:
                cached_feature_file_path = "{}/{}.json".format(self.cached_features_path, feature_name)
                predictor = self.qpp_dict[feature_name]()
                return CachedQPP(predictor, cached_feature_file_path)
            return self.qpp_dict[feature_name]()

        if self.cached_features_path is not None:
            cached_feature_file_path = "{}/{}.json".format(self.cached_features_path, feature_name)
            predictor = self.qpp_dict[feature_name](params)
            return CachedQPP(predictor, cached_feature_file_path, **params)
        return self.qpp_dict[feature_name](params)
