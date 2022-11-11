import pandas as pd
import json
import scipy.stats
import time
import argparse
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from experiments.qpp.qpp_feature_extraction import  QPPFeatureFactory
from experiments.qpp.qpp_utils import load_data,create_label_dict,create_ctx,topic_evaluate_extractor,evaluate_topic_predictor
from experiments.qpp.const import QPP_FEATURES_PARAMS



REWRITE_METHODS=['raw','t5','all','hqe','quretec','manual']
REWRITE_METHODS=['t5','all','hqe','quretec']
#REWRITE_METHODS=['t5']
REWRITE_METHODS=['all','quretec']
DEFAULT_RES_DIR="kld_100"
DEFAULT_VALID_DIR="valid_kld_100"
DEFAULT_RES_DIR="rerank_kld_100"
DEFAULT_VALID_DIR="rerank_valid_kld_100"
DEFAULT_COL = "or_quac"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
res_file_name="pre_ret_qpp.json"
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq"]

DEFAULT_QUERY_FIELD="first_stage_rewrites"
DEFAULT_QUERY_FIELD="second_stage_queries"

def evaluate_threshold(t_val,features,labels):
    qids=list(labels.keys())
    q_labels= [1 if labels[qid]>0 else 0 for qid in qids]
    q_pred= [1 if features[qid]>=t_val else 0 for qid in qids]
    return accuracy_score(q_labels,q_pred)

def set_threshold(features,labels):
    threshold_acc=[(t_val,evaluate_threshold(t_val,features,labels)) for t_val in features.values()]
    best_threshold=max(threshold_acc,key=lambda x:x[1])
    print(best_threshold)
    return best_threshold[0]


if __name__=="__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric",default="recip_rank")
    parser.add_argument("--col",default=DEFAULT_COL)
    parser.add_argument("--res_dir",default=DEFAULT_RES_DIR)
    parser.add_argument("--valid_dir", default=DEFAULT_VALID_DIR)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--query_rewrite_field",default=DEFAULT_QUERY_FIELD)
    parser.add_argument("--cache_results",action='store_true',default=False)
    parser.add_argument("--load_cached_feature",action='store_true',default=False)
    parser.add_argument("--per_turn_tunning",action='store_true',default=False)
    parser.add_argument("--calc_threshold_predictor",action='store_true',default=False)
    parser.add_argument("--oracle_tunning",action='store_true',default=False)
    args=parser.parse_args()
    metric = args.metric
    col=args.col
    res_dir=args.res_dir
    valid_dir=args.valid_dir
    cache_results=args.cache_results
    load_cached_feature=args.load_cached_feature
    qpp_res_dir_base=args.qpp_res_dir_base
    selected_features=args.features
    query_rewrite_field=args.query_rewrite_field
    calc_threshold_predictor=args.calc_threshold_predictor
    per_turn_tunning=args.per_turn_tunning
    oracle_tunning=args.oracle_tunning
    EVAL_VALID_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(valid_dir, col)
    RUNS_VALID_PATH = "/v/tomergur/convo/res/{}/{}".format(valid_dir, col)
    runs_valid, rewrites_valid, rewrites_eval_valid, turns_text_valid = load_data(REWRITE_METHODS, EVAL_VALID_PATH, RUNS_VALID_PATH, query_rewrite_field,
                                                          col)
    label_dict_valid = create_label_dict(rewrites_eval_valid, metric)
    print("loaded valid data")
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    runs, rewrites, rewrites_eval, turns_text = load_data(REWRITE_METHODS, EVAL_PATH, RUNS_PATH, query_rewrite_field,
                                                          col)
    label_dict = create_label_dict(rewrites_eval, metric)
    print("loaded test data")
    sids = rewrites_eval[REWRITE_METHODS[0]].sid.unique()
    qpp_res_dir = "{}/{}/{}/".format(qpp_res_dir_base, res_dir, col)
    qpp_factory = QPPFeatureFactory(col,qpp_res_dir+"/valid_cache/" if load_cached_feature else None)
    corr_res = {}
    threshold_res = {"baseline":{}}
    ctx_valid = create_ctx(runs_valid, rewrites_valid, turns_text_valid,col)
    ctx = create_ctx(runs, rewrites, turns_text,col)
    # initial scatter code here
    for feature in selected_features:
        print("start calculate feature:",feature)
        corr_res[feature] = {}
        threshold_res[feature]={}
        corr_raw_res = {}
        hp_configs = QPP_FEATURES_PARAMS.get(feature, {})
        print("num configs", len(hp_configs))
        # scatter code can be inserted here
        valid_features_cache = {}
        feature_cache={}
        for method_name, method_rewrites in rewrites_valid.items():
            start_time = time.time()
            method_runs = runs_valid[method_name]
            method_ctx = ctx_valid[method_name]
            labels = label_dict_valid[method_name]

            method_runs_test = runs[method_name]
            labels_test=label_dict[method_name]
            method_ctx_test = ctx[method_name]
            method_rewrites_test=rewrites[method_name]
            if oracle_tunning:
                print("oracle!!!!!!!!!!!!!!!")
                method_ctx=method_ctx_test
                method_rewrites=method_rewrites_test
                labels=labels_test

            # run all scores:
            feature_calc_start_time = time.time()
            extractors = [qpp_factory.create_qpp_extractor(feature, **hp_config) for hp_config in hp_configs] if len(
                hp_configs) > 0 else [qpp_factory.create_qpp_extractor(feature)]
            feature_val_valid = [topic_evaluate_extractor(extractor, method_rewrites, labels, method_ctx, True) for extractor in
                           extractors]

            print([(x, y[0]) for x, y in zip(hp_configs, feature_val_valid)])
            print("valid feature calc time:", time.time() - feature_calc_start_time)

            if per_turn_tunning and len(hp_configs) > 0:
                values_valid = [v[1] for v in feature_val_valid]
                qids = list(labels_test.keys())
                split_token = "#" if len(qids[0].split("#")) > 1 else "_"
                turns_labels = {}
                for qid in qids:
                    cur_tid = qid.split(split_token)[1]
                    if cur_tid not in turns_labels:
                        turns_labels[cur_tid] = {}
                    turns_labels[cur_tid][qid] = labels_test[qid]
                features_val_test={}
                for tid,turn_labels in turns_labels.items():
                    print(tid,"num test labels:",len(turn_labels))
                    print(tid, "num valid labels:", len({qid: v for qid, v in method_rewrites.items() if qid.split(split_token)[1] == tid}))
                    valid_turn_corr=[evaluate_topic_predictor(v,labels,"sturn_{}_kendall".format(tid)) for v in values_valid]
                    print([(x, y) for x, y in zip(hp_configs, valid_turn_corr)])
                    valid_selected_hp = np.argmax([abs(x) for x in valid_turn_corr])
                    selected_extractor = extractors[valid_selected_hp]
                    method_turn_rewrites = {qid: v for qid, v in method_rewrites_test.items() if qid.split(split_token)[1] == tid}
                    _, turn_val_test = topic_evaluate_extractor(selected_extractor, method_turn_rewrites,turn_labels, method_ctx_test, True,"TPA")
                    features_val_test.update(turn_val_test)
                test_corr=evaluate_topic_predictor(features_val_test,labels_test,"turn_kendall")
            else:
                corr_valid = [v[0] for v in feature_val_valid]
                valid_selected_hp = np.argmax([abs(x) for x in corr_valid])
                if len(hp_configs) > 0:
                    print(hp_configs[valid_selected_hp])
                selected_extractor = extractors[valid_selected_hp]
                test_corr, features_val_test = topic_evaluate_extractor(selected_extractor, method_rewrites_test,
                                                                        labels_test, method_ctx_test, True)
            corr_res[feature][method_name] = round(test_corr,3)
            if cache_results:
                if len(hp_configs) == 0:
                    valid_features_cache[method_name] = feature_val_valid[0][1]
                else:
                    method_cache = []
                    for hp_config, f_val in zip(hp_configs, feature_val_valid):
                        method_cache.append((list(hp_config.items()), f_val[1]))
                    valid_features_cache[method_name] = method_cache
                feature_cache[method_name]= features_val_test
            #TODO: remove dead code
            if calc_threshold_predictor:
                predictor_start_time=time.time()
                threshold=set_threshold(feature_val_valid[0][1],labels)
                t_res=evaluate_threshold(threshold,features_val_test,labels_test)
                threshold_res[feature][method_name] = round(t_res,2)
                num_res=len(labels_test)
                num_label_1=len([x  for x in labels_test.values() if x>0])
                maj_vote=max(num_label_1,num_res-num_label_1)
                threshold_res["baseline"][method_name]=round(maj_vote/num_res,2)
                print("test acc res:",t_res,"baseline:",maj_vote/num_res)
                print("num results",num_res,"num 1 label:",num_label_1)
                print("predictor calc time:",time.time()-predictor_start_time)
        if cache_results:
            feature_name=feature if not oracle_tunning else feature+"_oracle"
            res_path = "{}/cache/{}{}_{}.json".format(qpp_res_dir, feature_name,"_pt" if per_turn_tunning else "", metric)
            with open(res_path, 'w') as f:
                json.dump(feature_cache,f)
            if oracle_tunning:
                continue
            valid_res_path = "{}/valid_cache/{}.json".format(qpp_res_dir, feature_name)
            with open(valid_res_path, 'w') as f:
                json.dump(valid_features_cache,f)


    r_res = []
    for feature, corr_vals in corr_res.items():
        cur_row = {"predictor": feature}
        cur_row.update(corr_vals)
        r_res.append(cur_row)
        row_df=pd.DataFrame([cur_row])
        feature_name = feature if not oracle_tunning else feature + "_oracle"
        res_path="{}/corr_{}{}_{}.csv".format(qpp_res_dir,feature_name,"_pt" if per_turn_tunning else "",metric)
        row_df.to_csv(res_path,index=False)
    r_df = pd.DataFrame(r_res)
    print(r_df)
    if calc_threshold_predictor:
        th_res=[]
        for feature, corr_vals in threshold_res.items():
            cur_row = {"predictor": feature}
            cur_row.update(corr_vals)
            th_res.append(cur_row)
            row_df = pd.DataFrame([cur_row])
            res_path = "{}/threshold_{}_{}.csv".format(qpp_res_dir, feature, metric)
            row_df.to_csv(res_path, index=False)

        threshold_df = pd.DataFrame(th_res)
        print(threshold_df)






