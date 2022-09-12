import pandas as pd
import json
import scipy.stats
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math

from .qpp_feature_extraction import QPPFeatureFactory
from .qpp_utils import load_data,create_label_dict,create_ctx
REWRITE_METHODS=['t5','all','hqe','quretec']
#REWRITE_METHODS=['t5','hqe','quretec']
DEFAULT_RES_DIR="kld"
DEFAULT_COL = "cast19"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/rewrites_comp/"
res_file_name="pre_ret_qpp.json"
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq"]
QPP_FEATURES_PARAMS={"WIG":[{"k":v} for v in [5,10,25,50,100,250,500,1000]],
                     "clarity":[{"k":v} for v in [5,10,25,50,100,250,500,1000]],
                     "NQC":[{"k":v} for v in [5,10,25,50,100,250,500,1000]],
                     "WIG_norm": [{"k": v} for v in [5, 10, 25, 50, 100, 250, 500, 1000]],
                     "clarity_norm": [{"k": v} for v in [5, 10, 25, 50, 100, 250, 500, 1000]],
                     "NQC_norm": [{"k": v} for v in [5, 10, 25, 50, 100, 250, 500, 1000]],
                     }


def corr_rewrite_selection(feature_dict,label_dict):
    methods=list(label_dict.keys())
    qids=list(label_dict[methods[0]].keys())
    corr_values=[]
    for qid in qids:
        q_feature=[feature_dict[m][qid] for m in methods]
        q_label = [label_dict[m][qid] for m in methods]
        if max(q_label) == min(q_label):
            continue
        corr,p_val=scipy.stats.pearsonr(q_feature,q_label)
        corr_values.append(corr)
    return np.mean(corr_values)

def evaluate_rewrite_selection(feature_dict,label_dict):
    methods=list(label_dict.keys())
    qids=list(label_dict[methods[0]].keys())
    selected_label=[]
    for qid in qids:
        highest_feature=max([(m,feature_dict[m][qid]) for m in methods],key=lambda x:x[1])
        selected_method=highest_feature[0]
        selected_label.append(label_dict[selected_method][qid])
    return np.mean(selected_label)

def evaluate_extractor(extractor, train_rewrites,train_labels, ctx,return_feature_values=False):
    feature_dict={}
    for method_name,method_rewrites in train_rewrites.items():
        method_ctx=ctx[method_name]
        feature_vals={qid:extractor.calc_qpp_feature(q,**method_ctx[qid]) for qid,q in method_rewrites.items()}
        feature_dict[method_name]=feature_vals
    if return_feature_values:
        return corr_rewrite_selection(feature_dict,train_labels),feature_dict
    return corr_rewrite_selection(feature_dict,train_labels)

def hp_feature_extractor(qpp_factory,feature_name,hp_configs,train_rewrites,train_labels,ctx,return_feature_values=False):
    if hp_configs=={}:
        extractor = qpp_factory.create_qpp_extractor(feature_name)
        if return_feature_values:
            _,feature_values=evaluate_extractor(extractor, train_rewrites, train_labels, ctx, True)
            return extractor,feature_values
        return extractor
    start_time=time.time()
    extractors = [qpp_factory.create_qpp_extractor(feature_name,**hp_config) for hp_config in hp_configs]
    if return_feature_values:
        evaluators_res=[evaluate_extractor(extractor, train_rewrites,train_labels, ctx,True) for extractor in extractors]
        train_res=[abs(v[0]) for v in evaluators_res]
        best_config = np.argmax(train_res)
        print(hp_configs[best_config])
        print("train time", time.time() - start_time)
        return extractors[best_config],evaluators_res[best_config][1]

    train_res = [abs(evaluate_extractor(extractor, train_rewrites,train_labels, ctx) for extractor in extractors)]
    best_config = np.argmax(train_res)
    print(hp_configs[best_config])
    print("train time",time.time()-start_time)
    return extractors[best_config]

def feature_per_query_eval(feature_dict,label_dict):
    methods = list(label_dict.keys())
    qids = list( label_dict[methods[0]].keys())
    corr_res=[]
    for qid in qids:
        label_lst=[label_dict[m][qid] for m in methods]
        feature_lst = [feature_dict[m][qid] for m in methods]
        #print(feature_lst,label_lst)
        corr,p_val=scipy.stats.pearsonr(feature_lst,label_lst)
        #print(corr)
        if math.isnan(corr):
            continue
        corr_res.append(corr)
    return np.mean(corr_res)

def feature_per_query_selection_eval(feature_dict,rewrite_eval):
    res=None
    methods=list(feature_dict.keys())
    qids=list(feature_dict[methods[0]].keys())
    for qid in qids:
        selected_rewrite=max([(m,feature_dict[m][qid]) for m in methods],key=lambda x:x[1])
        selected_method=selected_rewrite[0]
        selected_method_results=rewrite_eval[selected_method]
        q_res=selected_method_results[selected_method_results.qid==qid]
        res=res.append(q_res) if res is not None else q_res
    print(res.groupby("metric").mean())




def feature_pairwise_acc(feature_dict,label_dict):
    total_compares = 0
    good_selection = 0
    methods = list(label_dict.keys())
    qids = list( label_dict[methods[0]].keys())
    rewrites_pairwise_acc={}
    for method1, method2 in itertools.combinations(methods, 2):
        total_pair_compares=0
        good_selection_in_pair=0
        method1_label = label_dict[method1]
        method1_feature = feature_dict[method1]
        method2_label = label_dict[method2]
        method2_feature = feature_dict[method2]
        for qid in qids:
            if method1_label[qid] == method2_label[qid]:
                continue
            total_compares += 1
            total_pair_compares+=1
            best_rewrite = np.argmax([method1_label[qid], method2_label[qid]])
            highest_feature = np.argmax([method1_feature[qid], method2_feature[qid]])
            if best_rewrite == highest_feature:
                good_selection += 1
                good_selection_in_pair+=1
        rewrites_pairwise_acc[(method1,method2)]=(total_pair_compares,good_selection_in_pair,good_selection_in_pair/total_pair_compares)

    return total_compares,good_selection,rewrites_pairwise_acc

def rewrite_selection_eval(features_dict,label_dict,rewrite_eval,qpp_res_dir):
    r_df=[]
    for feature,feature_dict in features_dict.items():
        print("feature eval:",feature)
        feature_dir = "{}/{}".format(qpp_res_dir, feature)
        total_compares, good_selection,rewrites_pairwise_acc=feature_pairwise_acc(feature_dict,label_dict)
        #print(total_compares,good_selection,good_selection/total_compares)
        pairwise_res=[{"method1":"all comp","method2":"all comp","#comapres":total_compares,
                      "#correct":good_selection,"pairwise acc.":round(good_selection/total_compares,2)}]
        for method_pair,method_res in rewrites_pairwise_acc.items():
            pair_row={"method1":method_pair[0],"method2":method_pair[1]}
            pair_row["#comapres"]=method_res[0]
            pair_row["#correct"]=method_res[1]
            pair_row["pairwise acc."]=round(method_res[2],2)
            pairwise_res.append(pair_row)
        pairwise_df=pd.DataFrame(pairwise_res)
        print(pairwise_df)
        pairwise_df.to_csv("{}/pairwise_acc.csv".format(feature_dir),index=False)
        r=feature_per_query_eval(feature_dict,label_dict)
        feature_per_query_selection_eval(feature_dict, rewrite_eval)
        r_df.append({"feature":feature,"r":round(r,3)})
        corr_df=pd.DataFrame([{"feature":feature,"r":round(r,3)}])
        corr_df.to_csv("{}/corr_df.csv".format(feature_dir),index=False)

        #print("r val:",r)
    r_df=pd.DataFrame(r_df)
    print(r_df)


def write_feature_values(rewrite_methods,train_features,output_file):
    qids = list(train_features[rewrite_methods[0]].keys())
    rows = []
    for qid in qids:
        q_row = {"qid": qid}
        q_row.update({m: train_features[m][qid] for m in rewrite_methods})
        rows.append(q_row)
    train_res_df = pd.DataFrame(rows)
    train_res_df.to_csv(output_file, index=False)

def create_scatter_plot(features,labels,path,feature_name,metric):
    plt.figure(figsize=(12, 9), dpi=200)
    method_names=list(features.keys())
    qids=list(labels[method_names[0]].keys())
    for m in method_names:
        x=[features[m][qid] for qid in qids]
        y=[labels[m][qid] for qid in qids]
        plt.scatter(x,y,label=m)
    plt.xlabel(feature_name.replace("_"," "))
    plt.ylabel(metric.replace("_"," "))
    plt.legend()
    plt.title("scatter plot: {} values comapred to {} values".format(feature_name.replace("_"," "),metric.replace("_"," ")))
    plt.savefig(path)
    plt.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric",default="ndcg_cut_3")
    parser.add_argument("--col",default=DEFAULT_COL)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--folds_file_name",default="folds_10.json")
    parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--query_rewrite_field",default="first_stage_rewrites")
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    args=parser.parse_args()
    metric = args.metric
    col=args.col
    res_dir=args.res_dir
    qpp_res_dir_base=args.qpp_res_dir_base
    folds_file_name=args.folds_file_name
    selected_features=args.features
    query_rewrite_field=args.query_rewrite_field
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    runs, rewrites, rewrites_eval,turns_text = load_data(REWRITE_METHODS, EVAL_PATH, RUNS_PATH,query_rewrite_field,col)
    label_dict = create_label_dict(rewrites_eval, metric)
    qpp_res_dir = "{}/{}/{}/".format(qpp_res_dir_base,res_dir, col)
    with open("{}/{}".format(qpp_res_dir,folds_file_name)) as f:
        folds=json.load(f)
    #output files
    for feature in selected_features:
        feature_dir="{}/{}".format(qpp_res_dir,feature)
        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)
        feature_path={}
    qpp_factory = QPPFeatureFactory()
    ctx = create_ctx(runs, rewrites, turns_text)
    features_dict={}
    for feature in selected_features:
        feature_dir = "{}/{}".format(qpp_res_dir, feature)
        features_dict[feature]={m:{} for m in REWRITE_METHODS}

        for i,fold in enumerate(folds,start=1):
            train_sids=[str(sid) for sid in fold["train"]]
            valid_sids = [str(sid) for sid in fold["valid"]]
            test_sids=[str(sid) for sid in fold["test"]]
            train_rewrites,valid_rewrites,test_rewrites={},{},{}
            hp_configs = QPP_FEATURES_PARAMS.get(feature, {})
            train_labels,valid_labels,test_labels={},{},{}
            for method_name,method_rewrites in rewrites.items():
                method_runs=runs[method_name]
                train_rewrites[method_name]={qid:v for qid,v in method_rewrites.items() if qid.split("_")[0] in train_sids}
                valid_rewrites[method_name] = {qid: v for qid, v in method_rewrites.items() if
                                              qid.split("_")[0] in valid_sids}
                test_rewrites[method_name] = {qid: v for qid, v in method_rewrites.items() if
                                               qid.split("_")[0] in test_sids}
                label_method=label_dict[method_name]
                train_labels[method_name] = {qid: v for qid, v in label_method.items() if
                                               qid.split("_")[0] in train_sids}
                test_labels[method_name] = {qid: v for qid, v in label_method.items() if
                                              qid.split("_")[0] in test_sids}
                valid_labels[method_name] = {qid: v for qid, v in label_method.items() if
                                            qid.split("_")[0] in valid_sids}
            extractor,train_features=hp_feature_extractor(qpp_factory,feature,hp_configs,train_rewrites,train_labels,ctx,True)
            #output feature values
            features_values_path="{}/train_fold_{}_scores.csv".format(feature_dir, i)
            write_feature_values(REWRITE_METHODS,train_features,features_values_path)
            test_features,valid_features={},{}
            for method_name,method_test_rewrites in test_rewrites.items():
                method_ctx=ctx[method_name]
                test_features[method_name]={qid:extractor.calc_qpp_feature(query, **method_ctx[qid]) for qid,query in method_test_rewrites.items()}
                features_dict[feature][method_name].update(test_features[method_name])
                method_valid_rewrites=valid_rewrites[method_name]
                valid_features[method_name]={qid:extractor.calc_qpp_feature(query, **method_ctx[qid]) for qid,query in method_valid_rewrites.items()}
            test_features_values_path = "{}/test_fold_{}_scores.csv".format(feature_dir, i)
            write_feature_values(REWRITE_METHODS, test_features, test_features_values_path)
            valid_features_values_path = "{}/valid_fold_{}_scores.csv".format(feature_dir, i)
            write_feature_values(REWRITE_METHODS, valid_features, valid_features_values_path)


    for feature,feature_vals in features_dict.items():
        feature_dir = "{}/{}".format(qpp_res_dir, feature)
        features_plot_path = "{}/scatter_plot.png".format(feature_dir)
        create_scatter_plot(feature_vals, label_dict, features_plot_path, feature, metric)
    rewrite_selection_eval(features_dict, label_dict, rewrites_eval,qpp_res_dir)


