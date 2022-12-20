from dataclasses import dataclass
import json
import itertools
import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse

COLLECTIONS = ['or_quac', 'topiocqa']
DEFAULT_REWRITE_METHODS = ['all', 'quretec', 't5', 'hqe']
DEFAULT_FEATURES=["bert_qpp","ref_hist_agg_bert_qpp","ref_rewrites_agg_bert_qpp","ref_combined_bert_qpp","bert_pl","ref_hist_agg_bert_pl","ref_rewrites_agg_bert_pl","ref_combined_bert_pl","many_turns_bert_qpp_tokens","ref_rewrites_agg_many_turns_bert_qpp_tokens","rewrites_bert_qpp","ref_hist_agg_rewrites_bert_qpp"]

METHOD_DISPLAY_NAME={"WIG_norm":"WIG","clarity_norm":"clarity","NQC_norm":"NQC","bert_qpp":"Bert QPP",
                     "WIG_norm_pt":"WIG -HP per turn","clarity_norm_pt":"clarity -HP per turn",
                     "NQC_norm_pt":"NQC -HP per turn","bert_qpp_pt":"Bert QPP -HP per turn",
                     "st_bert_qpp_pt":"Bert QPP - fine tuned and HP per turn",
                     "st_bert_qpp_oracle_pt":"BERT QPP - fine tuned and HP per turn(HP selected by oracle)",
                     "bert_qpp_oracle": "BERT QPP - (HP selected by oracle for all turns)",
                     "bert_qpp_cls":"Bert QPP","bert_qpp_reg":"Bert QPP(MSE loss)",
                     "bert_qpp_or_quac":"Bert QPP fine-tuned on Or QUAC",
                     "bert_qpp_topiocqa":"Bert QPP fine-tuned on TopioCQA",
                     "bert_qpp_hist":"Bert QPP+ raw history","bert_qpp_hist_or_quac":"Bert QPP+history fine-tuned on Or QUAC",
                     "bert_qpp_hist_topiocqa":"Bert QPP+history fine-tuned on TopioCQA",
                     "bert_qpp_prev":"Bert QPP+previous queries",
                     "bert_pl": "Bert PL ",
                     "many_turns_bert_qpp":"dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens": "dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens_skturns": "dialogue groupwise QPP(vary history length)",
                     "many_turns_bert_qpp_online": "dialogue groupwise QPP - online inference",
                     "many_turns_bert_qpp_hist": "dialogue groupwise QPP+raw history",
                     "many_turns_bert_qpp_prev": "dialogue groupwise QPP+previous queries",
                     "seq_qpp":"dialogue LSTM QPP",
                     "rewrites_bert_qpp": "rewrites groupwise qpp",
                     "ref_hist_agg_rewrites_bert_qpp": "rewrites groupwise qpp - history REF RBO ",
                     "ref_rewrites_bert_qpp":"Bert QPP -rewrites REF RBO",
                     "ref_rewrites_agg_bert_qpp": "Bert QPP -rewrites REF RBO",
                     "ref_rewrites_agg_many_turns_bert_qpp_tokens": "dialogue groupwise QPP -rewrites REF RBO",
                     "ref_rewrites_agg_bert_pl": "Bert pl -rewrites REF RBO",
                     "ref_combined_bert_qpp":"Bert QPP - combined REF RBO",
                     "ref_combined_many_turns_bert_qpp_tokens": "dialogue groupwise QPP - combined REF RBO",
                     "ref_combined_bert_pl": "Bert pl - combined REF RBO",
                     "ref_rewrites_bert_qpp_all_methods": "Bert QPP -all rewrite REF RBO",
                     "ref_rewrites_bert_qpp_t5_methods": "Bert QPP -t5 rewrite REF RBO",
                     "ref_rewrites_bert_qpp_quretec_methods": "Bert QPP -quretec rewrite REF RBO",
                     "ref_rewrites_bert_qpp_hqe_methods": "Bert QPP -hqe rewrite REF RBO",
                     "ref_hist_bert_qpp_cls":"Bert QPP -REF RBO","many_turns_bert_qpp_cls":"dialogue groupwise QPP",
                     "ref_hist_bert_qpp": "Bert QPP - history REF RBO",
                     "ref_hist_agg_bert_qpp": "Bert QPP - history REF RBO",
                     "ref_hist_agg_many_turns_bert_qpp_tokens": "dialogue groupwise QPP - history REF RBO",
                     "ref_hist_agg_bert_pl": "Bert PL - history REF RBO",
                     "ref_hist_bert_qpp_skturns": "Bert QPP - history REF RBO(varying history size)",
                     "ref_hist_bert_qpp":"Bert QPP - history REF RBO","ref_hist_bert_qpp_pt":"Bert QPP - REF RBO, HP per turn"}


@dataclass
class CompResults:
    wins: int = 0
    ties: int = 0
    losses: int = 0
    sgni_wins: int = 0
    sgni_losses: int = 0

    def __add__(self, other):
        wins = self.wins + other.wins
        ties = self.ties + other.ties
        losses = self.losses + other.losses
        sgni_wins = self.sgni_wins + other.sgni_wins
        sgni_losses = self.sgni_losses + other.sgni_losses
        return CompResults(wins, ties, losses, sgni_wins, sgni_losses)


def load_feature_expr(col, features,qpp_eval_metric):
    res = {}
    for feature in features:
        feature_eval = {}
        print("calc feature:", feature)
        exp_path = "{}/{}/{}/exp_per_turn_{}_{}_{}_30_{}.json".format(qpp_res_dir_base, res_dir, col,
                                                                           qpp_eval_metric,feature, metric, subsamples_size)
        with open(exp_path) as f:
            exp_turns = json.load(f)
        res[feature] = exp_turns
    return res


def t_test_paired(new_method, baseline, alpha=.05):
    t, p_val = stats.ttest_rel(new_method, baseline, alternative="greater")
    if p_val < alpha:
        assert (t > 0)
    return t > 0 and p_val < alpha


def create_comp_res(feature_eval, feature_eval2, split_eval, split_eval2):
    wins = sum([1 for x, y in zip(feature_eval, feature_eval2) if x > y])
    ties = sum([1 for x, y in zip(feature_eval, feature_eval2) if math.isclose(x, y)])
    losses = sum([1 for x, y in zip(feature_eval, feature_eval2) if x < y])
    sgni_wins = sum([1 for x, y in zip(split_eval, split_eval2) if t_test_paired(x, y)])
    sgni_losses = sum([1 for x, y in zip(split_eval, split_eval2) if t_test_paired(y, x)])
    return CompResults(wins, ties, losses, sgni_wins, sgni_losses)


def display_comp_res(features, feature_comps,output_path):
    data = []
    for feature in features:
        feature_row = {"name": METHOD_DISPLAY_NAME.get(feature,feature.replace("_"," "))}
        for feature2 in features:
            feature2_display = METHOD_DISPLAY_NAME.get(feature2,feature2.replace("_"," "))
            if feature == feature2:
                feature_row[feature2_display] = "---"
            else:
                comp_res = feature_comps[feature, feature2]
                res_text = "{}({})/{}/{}({})".format(comp_res.wins, comp_res.sgni_wins, comp_res.ties, comp_res.losses,
                                                      comp_res.sgni_losses)
                feature_row[feature2_display]=res_text
        data.append(feature_row)
    res_table=pd.DataFrame(data)
    pd.set_option('display.max_columns', None)
    print(res_table)
    if output_path is not None:
        res_table.to_csv(output_path,index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--cols", nargs='+', default=COLLECTIONS)
    parser.add_argument("--subsamples_size", type=int,default=50)
    parser.add_argument("--output_path",default=None)
    parser.add_argument("--qpp_eval_metric",default="kendall")
    parser.add_argument("--features",nargs='+',default=DEFAULT_FEATURES)
    parser.add_argument("--rewrite_methods",nargs='+',default=DEFAULT_REWRITE_METHODS)
    args=parser.parse_args()
    collections=args.cols
    subsamples_size = args.subsamples_size
    output_path=args.output_path
    qpp_eval_metric=args.qpp_eval_metric
    FEATURES=args.features
    REWRITE_METHODS=args.rewrite_methods
    res_dir = "rerank_kld_100"
    qpp_res_dir_base = "/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
    metric = "recip_rank"

    feature_comps = {(f1, f2): CompResults() for f1, f2 in itertools.permutations(FEATURES, 2)}
    for col in collections:
        EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
        RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
        sep_token = "#" if col == "or_quac" else "_"
        features_exp = load_feature_expr(col, FEATURES,qpp_eval_metric)
        for rewrite_method in REWRITE_METHODS:
            print("calc res for:", rewrite_method)
            features_eval = {f: [] for f in FEATURES}
            splits_eval = {f: [] for f in FEATURES}
            for feature in FEATURES:
                feature_splits_res = features_exp[feature][rewrite_method]
                max_turns = len(feature_splits_res[0])
                turns = range(max_turns)
                for i in turns:
                    turn_kendall = [feature_res[i] for feature_res in feature_splits_res]
                    turn_res = round(np.mean(turn_kendall), 3)
                    features_eval[feature].append(turn_res)
                    splits_eval[feature].append(turn_kendall)
            for feature_pairs, features_comp_res in feature_comps.items():
                f1, f2 = feature_pairs
                feature_eval1 = features_eval[f1]
                feature_eval2 = features_eval[f2]
                split_eval1 = splits_eval[f1]
                split_eval2 = splits_eval[f2]
                method_feature_comp = create_comp_res(feature_eval1, feature_eval2, split_eval1, split_eval2)
                feature_comps[feature_pairs] = features_comp_res + method_feature_comp
        print(feature_comps)
    display_comp_res(FEATURES,feature_comps,output_path)
