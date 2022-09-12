import pandas as pd
import json
from .qpp_feature_extraction import QPPFeatureExtractor
import itertools
import scipy.stats
import math
import time
import argparse
REWRITE_METHODS=['t5','all','hqe','quretec','manual']
RES_DIR="kld"
COL="cast19"
EVAL_PATH="/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(RES_DIR,COL)
RUNS_PATH="/v/tomergur/convo/res/{}/{}".format(RES_DIR,COL)
res_file_name="pre_ret_qpp.json"
import numpy as np
SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","WIG"]
QPP_FEATURES_PARAMS={"WIG":[{"k":v} for v in [5,10,25,100]]}
#,"max_var","avg_var"

def load_data(rewrite_methods,eval_path,runs_path,queries_field="first_stage_rewrites"):

    runs,rewrites,rewrites_eval={},{},{}
    for rewrite_method in rewrite_methods:
        input_path = "{}/{}.txt".format(eval_path, rewrite_method)
        eval_list = pd.read_csv(input_path, header=None, names=["metric", "qid", "value"], delimiter="\t")
        # eval_list.metric=eval_list.metric.str.strip()
        rewrites_eval[rewrite_method] = eval_list
    #second_stage_queries
    for rewrite_method in rewrite_methods:
        input_path = "{}/{}_queries.json".format(runs_path,rewrite_method)
        with open(input_path) as f:
            rewrite_json=json.load(f)
        rewrites[rewrite_method]={qid:q_dict[queries_field][0] for qid,q_dict in rewrite_json.items() }

    for rewrite_method in rewrite_methods:
        run_path= "{}/{}_run.txt".format(runs_path,rewrite_method)
        method_run=pd.read_csv(run_path, header=None, names=["qid", "Q0", "docid", "ranks", "score", "info"],
                           delimiter="\t")
        runs[rewrite_method]=method_run


    return  runs,rewrites,rewrites_eval


def create_label_dict(rewrites_eval,metric):
    return {k: df[(df.metric.str.startswith(metric))&(~(df.qid=="all"))].set_index("qid").value.to_dict() for k, df in rewrites_eval.items()}

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

def rewrite_selection_eval(features_dict,label_dict,rewrite_eval):
    for feature,feature_dict in features_dict.items():
        print("feature eval:",feature)
        total_compares, good_selection,rewrites_pairwise_acc=feature_pairwise_acc(feature_dict,label_dict)
        print(total_compares,good_selection,good_selection/total_compares)
        print(rewrites_pairwise_acc)
        r=feature_per_query_eval(feature_dict,label_dict)
        feature_per_query_selection_eval(feature_dict, rewrite_eval)
        print("r val:",r)

def feature_topic_eval(feature_dict,label_dict):
    res= {}
    methods = list(label_dict.keys())
    qids = list(label_dict[methods[0]].keys())
    for method in methods:
        feature_lst = [feature_dict[method][qid] for qid in qids]
        label_lst = [label_dict[method][qid] for qid in qids]
        corr,p_val=scipy.stats.pearsonr(feature_lst,label_lst)
        res[method]=corr
    return res


def topic_comp_eval(features_dict,label_dict,rewrite_eval):
    r_res=[]
    for feature, feature_dict in features_dict.items():
        print("feature eval:", feature)
        r_vals=feature_topic_eval(feature_dict,label_dict)
        #print(r_vals)
        cur_row={"predictor":feature}
        cur_row.update(r_vals)
        r_res.append(cur_row)
    r_df=pd.DataFrame(r_res)
    print(r_df)


if __name__=="__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--topic_comp",action="store_true",default=True)
    parser.add_argument("--metric",default="ndcg_cut_3")
    args=parser.parse_args()
    topic_comp=args.topic_comp
    metric = args.metric
    runs, rewrites, rewrites_eval=load_data(REWRITE_METHODS,EVAL_PATH,RUNS_PATH)
    label_dict=create_label_dict(rewrites_eval,metric)
    features_dict={}
    qpp_feature_extractor=QPPFeatureExtractor()

    for feature in SELECTED_FEATURES:
        start_time=time.time()
        features_dict[feature]={}
        for method_name,method_rewrites in rewrites.items():
            method_runs=runs[method_name]
            ctx={qid:{"res_list":list(zip(q_run.docid.tolist(),q_run.docid.tolist()))} for qid,q_run in method_runs.groupby(['qid'])}
            if feature in QPP_FEATURES_PARAMS:
                labels=label_dict[method_name]
                qpp_feature_extractor.hp_tune_extractors({feature:QPP_FEATURES_PARAMS[feature]},method_rewrites,ctx,labels)
            method_res={qid:qpp_feature_extractor.extract_feature(feature,q,**ctx[qid]) for qid,q in method_rewrites.items()}
            features_dict[feature][method_name]=method_res
        print("calc feature:",feature,"time:",time.time()-start_time)
    if topic_comp:
        topic_comp_eval(features_dict,label_dict,rewrites_eval)
    else:
        rewrite_selection_eval(features_dict,label_dict,rewrites_eval)
    with open("{}/{}".format(RUNS_PATH,res_file_name),'w') as f:
        json.dump(features_dict,f)







