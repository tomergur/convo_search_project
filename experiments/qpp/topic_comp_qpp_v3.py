import random

import pandas as pd
import json
import scipy.stats
import time
import argparse
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from .const import QPP_FEATURES_PARAMS
from .qpp_feature_extraction import QPPFeatureFactory
from .cached_qpp import KeyValueQPP
from .ref_list_qpp import RefListQPP,CombinedRefListQPP
from .qpp_utils import load_data, create_label_dict, create_ctx, calc_topic_corr, topic_evaluate_extractor, \
    calc_topic_pairwise_acc, evaluate_topic_predictor

REWRITE_METHODS = ['raw', 't5', 'all', 'hqe', 'quretec', 'manual']
REWRITE_METHODS = ['t5', 'all', 'hqe', 'quretec']
#REWRITE_METHODS = ['all', 'quretec']
#REWRITE_METHODS = ['t5', 'quretec']
#REWRITE_METHODS = ['all']
DEFAULT_RES_DIR = "rerank_kld_100"
DEFAULT_COL = "cast19"
DEFAULT_QPP_RES_DIR = "/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
res_file_name = "pre_ret_qpp.json"
DEFAULT_SELECTED_FEATURES = ["q_len", "max_idf", "avg_idf", "max_scq", "avg_scq"]

DEFAULT_QUERY_FIELD = "first_stage_rewrites"
DEFAULT_QUERY_FIELD = "second_stage_queries"

RBO_N_VALS = [5, 10, 25, 50, 100]
LAMBD_VALS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1]
REF_RBO_CONFIG=list(itertools.product(RBO_N_VALS,LAMBD_VALS))
REF_COMBINED_RBO_CONFIG=list(itertools.product(RBO_N_VALS,LAMBD_VALS,LAMBD_VALS))
def feature_topic_eval(feature_dict, label_dict):
    res = {}
    methods = list(label_dict.keys())
    for method in methods:
        corr = calc_topic_corr(feature_dict[method], label_dict[method])
        res[method] = corr
    return res


def topic_comp_eval(features_dict, label_dict, rewrite_eval):
    r_res = []
    for feature, feature_dict in features_dict.items():
        print("feature eval:", feature)
        r_vals = feature_topic_eval(feature_dict, label_dict)
        # print(r_vals)
        cur_row = {"predictor": feature}
        cur_row.update(r_vals)
        r_res.append(cur_row)
    r_df = pd.DataFrame(r_res)
    print(r_df)


def load_or_create_splits(sids, num_splits, qpp_res_dir):
    splits_file_name = "{}/splits_{}.json".format(qpp_res_dir, num_splits)
    if os.path.isfile(splits_file_name):
        with open(splits_file_name) as f:
            splits = json.load(f)
        return splits
    splits = []
    subsamples = []
    fold_size = int(len(sids) / 2)
    for i in range(num_splits):
        perm_sessions = np.random.permutation(sids).tolist()
        train_fold = perm_sessions[:fold_size]
        test_fold = perm_sessions[fold_size:]
        print(train_fold, test_fold, len(train_fold), len(test_fold))
        splits.append([train_fold, test_fold])
    with open(splits_file_name, 'w') as f:
        json.dump(splits, f)
    return splits

def load_or_create_smart_subsamples(qids, splits, qpp_res_dir,rewrites_labels, subsample_size=50, max_turn=10,zero_prec=.1):
    subsamples_file_name = "{}/subsamples_smart_{}_{}_{}.json".format(qpp_res_dir, len(splits), subsample_size, max_turn)
    if os.path.isfile(subsamples_file_name):
        with open(subsamples_file_name) as f:
            subsamples = json.load(f)
        return subsamples
    subsamples = {}
    split_token = "#" if len(qids[0].split("#")) > 1 else "_"
    tids = list(set([qid.split(split_token)[1] for qid in qids]))
    tids = [int(x) for x in tids]
    tids.sort()
    print("tids vals:", tids)
    t_qids = {tid: [] for tid in tids}
    for qid in qids:
        _, tid = qid.split(split_token)
        tid = int(tid)
        t_qids[tid].append(qid)
    tids = tids[:max_turn]
    num_zeroes=int(subsample_size*zero_prec)
    for rewrite_method,labels in rewrites_labels.items():
        method_susamples=[]
        for split in splits:
            fold1, fold2 = split
            fold1_subsamples, fold2_subsamples = [], []
            for tid in tids:
                qids_fold1 = [qid for qid in t_qids[tid] if qid.split(split_token)[0] in fold1]
                print("len qids fold 1", tid, len(qids_fold1))
                zero_res_qids_fold1=[qid for qid in qids_fold1 if labels[qid]==0]
                non_zero_res_qids_fold1=[qid for qid in qids_fold1 if labels[qid]>0]
                print("num of non zero qids fold1:",tid,len(non_zero_res_qids_fold1))
                fold1_subsamples += random.sample(zero_res_qids_fold1, num_zeroes) if len(
                    zero_res_qids_fold1) >= num_zeroes else zero_res_qids_fold1
                fold1_subsamples += random.sample(non_zero_res_qids_fold1, subsample_size-num_zeroes) if len(
                    non_zero_res_qids_fold1) >= subsample_size-num_zeroes else non_zero_res_qids_fold1

                qids_fold2 = [qid for qid in t_qids[tid] if qid.split(split_token)[0] in fold2]
                zero_res_qids_fold2=[qid for qid in qids_fold2 if labels[qid]==0]
                non_zero_res_qids_fold2=[qid for qid in qids_fold2 if labels[qid]>0]
                fold2_subsamples += random.sample(zero_res_qids_fold2, num_zeroes) if len(
                    zero_res_qids_fold2) >= num_zeroes else zero_res_qids_fold2
                fold2_subsamples += random.sample(non_zero_res_qids_fold2, subsample_size-num_zeroes) if len(
                    non_zero_res_qids_fold2) >= subsample_size-num_zeroes else non_zero_res_qids_fold2

            method_susamples.append([fold1_subsamples, fold2_subsamples])
        subsamples[rewrite_method]=method_susamples
    with open(subsamples_file_name, 'w') as f:
        json.dump(subsamples, f)
    return subsamples

def load_or_create_subsamples(qids, splits, qpp_res_dir, subsample_size=50, max_turn=10):
    subsamples_file_name = "{}/subsamples_{}_{}_{}.json".format(qpp_res_dir, len(splits), subsample_size, max_turn)
    if os.path.isfile(subsamples_file_name):
        with open(subsamples_file_name) as f:
            subsamples = json.load(f)
        return subsamples
    subsamples = []
    split_token = "#" if len(qids[0].split("#")) > 1 else "_"
    tids = list(set([qid.split(split_token)[1] for qid in qids]))
    tids = [int(x) for x in tids]
    tids.sort()
    print("tids vals:", tids)
    t_qids = {tid: [] for tid in tids}
    for qid in qids:
        _, tid = qid.split(split_token)
        tid = int(tid)
        t_qids[tid].append(qid)
    tids = tids[:max_turn]
    for split in splits:
        fold1, fold2 = split
        fold1_subsamples, fold2_subsamples = [], []
        for tid in tids:
            qids_fold1 = [qid for qid in t_qids[tid] if qid.split(split_token)[0] in fold1]
            print("len qids fold 1", tid, len(qids_fold1))
            fold1_subsamples += random.sample(qids_fold1, subsample_size) if len(
                qids_fold1) >= subsample_size else qids_fold1
            qids_fold2 = [qid for qid in t_qids[tid] if qid.split(split_token)[0] in fold2]
            fold2_subsamples += random.sample(qids_fold2, subsample_size) if len(
                qids_fold2) >= subsample_size else qids_fold2
        subsamples.append([fold1_subsamples, fold2_subsamples])
    with open(subsamples_file_name, 'w') as f:
        json.dump(subsamples, f)
    return subsamples


def hp_feature_extractor(qpp_factory, feature_name, hp_configs, train_rewrites, train_labels, ctx):
    if hp_configs == {}:
        return qpp_factory.create_qpp_extractor(feature_name)
    extractors = [qpp_factory.create_qpp_extractor(feature_name, **hp_config) for hp_config in hp_configs]
    train_res = [topic_evaluate_extractor(extractor, train_rewrites, train_labels, ctx) for extractor in extractors]
    best_config = np.argmax(train_res)
    print(hp_configs[best_config])
    return extractors[best_config]

def agg_extractor_res(agg_extractors,train_qids,test_qids,label_dict,rewrites,ctx,first_tid,max_turn,qpp_eval_metric,qpp_per_turn_metric_name):
    res,per_turn_res={},{}
    for method_name, method_rewrites in rewrites.items():
        method_ctx = ctx[method_name]
        labels = label_dict[method_name]
        train_labels = {qid: v for qid, v in labels.items() if qid in train_qids}
        train_rewrites= {qid: v for qid, v in method_rewrites.items() if qid in train_qids}
        agg_feature_val = [topic_evaluate_extractor(extractor, train_rewrites, train_labels, method_ctx, True,qpp_eval_metric) for extractor in agg_extractors]
        selected_hp = np.argmax([x[0] for x in agg_feature_val])
        test_rewrites = {qid: v for qid, v in method_rewrites.items() if qid in test_qids}
        test_labels = {qid: v for qid, v in labels.items() if qid in test_qids}
        test_res=topic_evaluate_extractor(agg_extractors[selected_hp], test_rewrites, test_labels, method_ctx, True,qpp_eval_metric)
        res[method_name]=test_res[0]
        per_turn_res[method_name]=[evaluate_topic_predictor(test_res[1], test_labels, "sturn_{}_{}".format(i,qpp_per_turn_metric_name))
                                        for i in range(first_tid, first_tid + max_turn)]
    return res,per_turn_res




if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--qpp_res_dir_base", default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--num_splits", type=int, default=30)
    parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--query_rewrite_field", default=DEFAULT_QUERY_FIELD)
    parser.add_argument("--cache_results", action='store_true', default=False)
    parser.add_argument("--load_cached_feature", action='store_true', default=False)
    parser.add_argument("--qpp_eval_metric", default="turn_kendall")
    parser.add_argument("--max_turn", type=int, default=10)
    parser.add_argument("--subsamples_size", type=int, default=50)
    parser.add_argument("--smart_subsamples",action='store_true',default=False)
    parser.add_argument("--per_turn_tunning", action='store_true', default=False)
    parser.add_argument("--oracle_tunning",action='store_true',default=False)
    parser.add_argument("--agg_type",default="ref_rewrites_agg")
    args = parser.parse_args()
    metric = args.metric
    col = args.col
    res_dir = args.res_dir
    cache_results = args.cache_results
    load_cached_feature = args.load_cached_feature
    qpp_res_dir_base = args.qpp_res_dir_base
    num_splits = args.num_splits
    selected_features = args.features
    query_rewrite_field = args.query_rewrite_field
    qpp_eval_metric = args.qpp_eval_metric
    max_turn = args.max_turn
    subsamples_size = args.subsamples_size
    per_turn_tunning = args.per_turn_tunning
    oracle_tunning=args.oracle_tunning
    smart_subsamples=args.smart_subsamples
    agg_type=args.agg_type
    assert agg_type in ["ref_combined","ref_hist_agg","ref_rewrites_agg"]
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    runs, rewrites, rewrites_eval, turns_text = load_data(REWRITE_METHODS, EVAL_PATH, RUNS_PATH, query_rewrite_field,
                                                          col)
    label_dict = create_label_dict(rewrites_eval, metric)


    sids = rewrites_eval[REWRITE_METHODS[0]].sid.unique()
    qids = rewrites_eval[REWRITE_METHODS[0]].qid.unique()
    qpp_res_dir = "{}/{}/{}/".format(qpp_res_dir_base, res_dir, col)
    splits = load_or_create_splits(sids, num_splits, qpp_res_dir)
    if smart_subsamples:
        splits_subsamples = load_or_create_smart_subsamples(qids, splits, qpp_res_dir,label_dict, subsamples_size, max_turn)
    else:
        splits_subsamples = load_or_create_subsamples(qids, splits, qpp_res_dir, subsamples_size, max_turn)
    qpp_factory = QPPFeatureFactory(col, qpp_res_dir + "/cache/" if load_cached_feature else None)

    corr_res = {}
    first_tid = 0 if col == 'or_quac' else 1
    split_token = "#" if col == "or_quac" else "_"
    per_turn_metric_name = "kendall" if "kendall" in qpp_eval_metric else "TPA"
    ctx = create_ctx(runs, rewrites, turns_text, col,REWRITE_METHODS)
    for feature in selected_features:
        corr_raw_res = {method_name:[] for method_name in REWRITE_METHODS }
        per_turn_corr_raw_res = {method_name:[] for method_name in REWRITE_METHODS }
        hp_configs = QPP_FEATURES_PARAMS.get(feature, {})
        print("num configs", len(hp_configs))
        features_cache = {}
        extractors = [qpp_factory.create_qpp_extractor(feature, **hp_config) for hp_config in hp_configs] if len(
            hp_configs) > 0 else [qpp_factory.create_qpp_extractor(feature)]
        selected_configs= {method_name:[] for method_name in REWRITE_METHODS }
        for method_name, method_rewrites in rewrites.items():
            start_time = time.time()
            method_runs = runs[method_name]
            method_ctx = ctx[method_name]
            corr_vals = []
            per_turn_corr_res = []
            labels = label_dict[method_name]

            # run all scores:
            feature_calc_start_time = time.time()
            feature_val = [topic_evaluate_extractor(extractor, method_rewrites, labels, method_ctx, True) for extractor
                           in extractors]
            ''''''
            if cache_results:
                if len(hp_configs) == 0:
                    features_cache[method_name] = feature_val[0][1]
                else:
                    method_cache = []
                    for hp_config, f_val in zip(hp_configs, feature_val):
                        method_cache.append((list(hp_config.items()), f_val[1]))
                    features_cache[method_name] = method_cache

            print([(x, y[0]) for x, y in zip(hp_configs, feature_val)])
            print("feature calc time:", time.time() - feature_calc_start_time)

            method_split=splits_subsamples[method_name] if smart_subsamples else splits_subsamples
            for i, split in enumerate(method_split):
                split_start_time = time.time()
                fold1_labels = {qid: v for qid, v in labels.items() if qid in split[0]}
                fold2_labels = {qid: v for qid, v in labels.items() if qid in split[1]}

                if per_turn_tunning:
                    res_fold_a = {}
                    res_fold_b = {}
                    for i in range(first_tid, first_tid + max_turn):
                        tid = str(i)
                        turn_features_values_fold_1 = [{qid: v for qid, v in hp_res[1].items() if
                                                        qid in split[0] and qid.split(split_token)[1] == tid} for hp_res
                                                       in feature_val]
                        turn_features_values_fold_2 = [{qid: v for qid, v in hp_res[1].items() if
                                                        qid in split[1] and qid.split(split_token)[1] == tid} for hp_res
                                                       in feature_val]
                        turn_fold_1_corr = [evaluate_topic_predictor(x, fold1_labels, "sturn_{}_kendall".format(i)) for x in
                                            turn_features_values_fold_1]
                        turn_fold_2_corr = [evaluate_topic_predictor(x, fold2_labels, "sturn_{}_kendall".format(i)) for
                                            x in turn_features_values_fold_2]

                        fold1_selected_hp = np.argmax(turn_fold_1_corr) if not oracle_tunning else np.argmax(turn_fold_2_corr)
                        res_fold_a.update(turn_features_values_fold_2[fold1_selected_hp])

                        fold2_selected_hp = np.argmax(turn_fold_2_corr) if not oracle_tunning else np.argmax(turn_fold_1_corr)
                        res_fold_b.update(turn_features_values_fold_1[fold2_selected_hp])

                    corr_fold_a=evaluate_topic_predictor(res_fold_a, fold2_labels, qpp_eval_metric)
                    corr_fold_b=evaluate_topic_predictor(res_fold_b, fold1_labels, qpp_eval_metric)
                else:
                    features_values_fold_1 = [{qid: v for qid, v in hp_res[1].items() if qid in split[0]} for hp_res in
                                              feature_val]
                    features_values_fold_2 = [{qid: v for qid, v in hp_res[1].items() if qid in split[1]} for hp_res in
                                              feature_val]

                    fold1_corr = [evaluate_topic_predictor(x, fold1_labels, qpp_eval_metric) for x in
                                  features_values_fold_1]
                    fold2_corr = [evaluate_topic_predictor(x, fold2_labels, qpp_eval_metric) for x in
                                  features_values_fold_2]

                    fold1_selected_hp = np.argmax(fold1_corr) if not oracle_tunning else np.argmax(fold2_corr)
                    #corr_fold_a = fold2_corr[fold1_selected_hp]
                    #res_fold_a = features_values_fold_2[fold1_selected_hp]

                    fold2_selected_hp = np.argmax(fold2_corr) if not oracle_tunning else np.argmax(fold1_corr)
                    #corr_fold_b = fold1_corr[fold2_selected_hp]
                    #res_fold_b = features_values_fold_1[fold2_selected_hp]
                    selected_configs[method_name].append((feature_val[fold1_selected_hp][1],feature_val[fold1_selected_hp][1]))

                    if len(hp_configs) > 0:
                        print(hp_configs[fold1_selected_hp])
                        print(hp_configs[fold2_selected_hp])
                '''
                per_turn_corr_fold_a = [evaluate_topic_predictor(res_fold_a, fold2_labels, "sturn_{}_{}".format(i,per_turn_metric_name))
                                        for i in range(first_tid, first_tid + max_turn)]
                per_turn_corr_fold_b = [evaluate_topic_predictor(res_fold_b, fold1_labels, "sturn_{}_{}".format(i,per_turn_metric_name))
                                        for i in range(first_tid, first_tid + max_turn)]

                per_turn_corr_res.append(per_turn_corr_fold_a)
                per_turn_corr_res.append(per_turn_corr_fold_b)

                corr_vals.append(corr_fold_a)
                corr_vals.append(corr_fold_b)
                '''
                print("prep split time", time.time() - split_start_time)
            #print(corr_vals)
            #print(feature, method_name, np.mean(corr_vals), len(corr_vals))
            #corr_res[feature][method_name] = round(np.mean(corr_vals), 3)
            #corr_raw_res[method_name] = corr_vals
            #per_turn_corr_raw_res[method_name] = per_turn_corr_res
            print("calc time:", time.time() - start_time)
        for i, split in enumerate(splits_subsamples):
            split_start_time = time.time()
            fold_1_qids, fold2_qids = split
            feature_cache = KeyValueQPP({k: v[i][0] for k, v in selected_configs.items()})
            feature_cache2 = KeyValueQPP({k: v[i][1] for k, v in selected_configs.items()})
            if agg_type=="ref_rewrites_agg":
                agg_extractors = [RefListQPP(feature_cache, "ref_rewrites", n=n, lambd=lambd) for n, lambd in
                              REF_RBO_CONFIG]
                agg_extractors2 = [RefListQPP(feature_cache2, "ref_rewrites", n=n, lambd=lambd) for n, lambd in
                               REF_RBO_CONFIG]
            elif agg_type=="ref_hist_agg":
                agg_extractors = [RefListQPP(feature_cache, "history", n=n, lambd=lambd) for n, lambd in
                              REF_RBO_CONFIG]
                agg_extractors2 = [RefListQPP(feature_cache2, "history", n=n, lambd=lambd) for n, lambd in
                               REF_RBO_CONFIG]
            else:
                agg_extractors = [CombinedRefListQPP(feature_cache, n=n, lambd=lambd,lambd2=lambd2) for n, lambd,lambd2 in
                              REF_COMBINED_RBO_CONFIG]
                agg_extractors2 = [CombinedRefListQPP(feature_cache, n=n, lambd=lambd,lambd2=lambd2) for n, lambd,lambd2 in
                              REF_COMBINED_RBO_CONFIG]

            res_a, per_turn_res_a = agg_extractor_res(agg_extractors, fold_1_qids, fold2_qids, label_dict, rewrites,
                                                      ctx, first_tid,
                                                      max_turn, qpp_eval_metric,per_turn_metric_name)
            res_b, per_turn_res_b = agg_extractor_res(agg_extractors2, fold2_qids, fold_1_qids, label_dict, rewrites,
                                                      ctx, first_tid,
                                                      max_turn, qpp_eval_metric,per_turn_metric_name)
            for method_name in REWRITE_METHODS:
                corr_raw_res[method_name] += [res_a[method_name], res_b[method_name]]
                per_turn_corr_raw_res[method_name] += [per_turn_res_a[method_name], per_turn_res_b[method_name]]
            print("split time:", time.time() - split_start_time)
        assert(len(corr_raw_res['t5'])==60)

        corr_res[feature] = {method_name: round(np.mean(method_raw_res),3) for method_name,method_raw_res in corr_raw_res.items() }

        # write corr results for stat sgni.
        run_name=agg_type+"_"+feature
        run_name =run_name+"_pt" if per_turn_tunning else run_name
        run_name = run_name + "_oracle" if oracle_tunning else run_name
        run_name=run_name + "_ssb" if smart_subsamples else run_name

        feature_r_vals_path = "{}/exp_{}_{}_{}_{}_{}.json".format(qpp_res_dir, qpp_eval_metric, run_name, metric,
                                                               num_splits,subsamples_size)
        with open(feature_r_vals_path, 'w') as f:
            json.dump(corr_raw_res, f)
        per_turn_feature_r_vals_path = "{}/exp_per_turn_{}_{}_{}_{}_{}.json".format(qpp_res_dir,per_turn_metric_name, run_name, metric,
                                                                                      num_splits,subsamples_size)
        with open(per_turn_feature_r_vals_path, 'w') as f:
            json.dump(per_turn_corr_raw_res, f)
        if cache_results:
            features_cache_path = "{}/cache/{}.json".format(qpp_res_dir, feature)
            with open(features_cache_path, 'w') as f:
                json.dump(features_cache, f)

    r_res = []
    for feature, corr_vals in corr_res.items():
        run_name = agg_type + "_" + feature
        run_name =run_name+"_pt" if per_turn_tunning else run_name
        run_name = run_name + "_oracle" if oracle_tunning else run_name
        run_name=run_name + "_ssb" if smart_subsamples else run_name
        cur_row = {"predictor": run_name}
        cur_row.update(corr_vals)
        r_res.append(cur_row)
        row_df = pd.DataFrame([cur_row])
        res_path = "{}/{}_{}_{}_{}_{}.csv".format(qpp_res_dir, qpp_eval_metric, run_name, metric, num_splits,subsamples_size)
        row_df.to_csv(res_path, index=False)
    r_df = pd.DataFrame(r_res)
    print(r_df)
