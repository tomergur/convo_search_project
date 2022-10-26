import argparse
import json
import os
import numpy as np
import pandas as pd

from ..qpp_utils import create_label_dict,load_eval,calc_topic_corr,calc_topic_pairwise_acc
from scipy.stats import pearsonr,kendalltau,spearmanr
import matplotlib.pyplot as plt

DEFAULT_COL='or_quac'
DEFAULT_RES_DIR="kld_100"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG","NQC","clarity","bert_qpp","bert_qpp_or_quac"]
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG_norm","NQC_norm","clarity_norm","bert_qpp"]
REWRITE_METHODS=['t5','all','hqe','quretec']
#REWRITE_METHODS=['t5']

METHOD_DISPLAY_NAME={"WIG_norm":"WIG","clarity_norm":"clarity","NQC_norm":"NQC","bert_qpp":"Bert QPP",
                     "bert_qpp_or_quac":"Bert QPP fine-tuned on Or QUAC",
                     "bert_qpp_topiocqa":"Bert QPP fine-tuned on TopioCQA",
                     "bert_qpp_hist":"Bert QPP+ raw history","bert_qpp_hist_or_quac":"Bert QPP+history fine-tuned on Or QUAC",
                     "bert_qpp_hist_topiocqa":"Bert QPP+history fine-tuned on TopioCQA",
                     "bert_qpp_prev":"Bert QPP+previous queries",
                     "many_turns_bert_qpp":"dialogue groupwise QPP",
                     "many_turns_bert_qpp_hist": "dialogue groupwise QPP+raw history",
                     "many_turns_bert_qpp_prev": "dialogue groupwise QPP+previous queries"}

def create_ret_turn_graph(ret_res,out_dir,metric):
    ret_file_name = "{}/turn_ret_performance_{}.png".format(out_dir, metric)
    plt.figure()
    #figsize=(18,10)
    for rewrite_method,method_ret_res in ret_res.items():
        plt.plot(range(1,1+len(method_ret_res)),method_ret_res,'o-',label=rewrite_method)
    plt.legend()
    plt.xlabel("turn number")
    METRIC_DISPLAY_NAMES = {'recall_100': 'recall@100', 'recip_rank': 'MRR@100', 'map_cut_100': "map@100"}
    plt.ylabel(METRIC_DISPLAY_NAMES.get(metric))
    plt.legend()
    plt.savefig(ret_file_name)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--features", nargs='*', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--corr_type",default="pearson")
    parser.add_argument("--min_turn_samples",type=int,default=0)
    parser.add_argument("--graph_name",default="graph")
    parser.add_argument("--write_to_csv",action="store_true",default=False)
    args=parser.parse_args()
    metric=args.metric
    col=args.col
    res_dir=args.res_dir
    features=args.features
    min_turn_samples = args.min_turn_samples
    qpp_res_dir_base = args.qpp_res_dir_base
    graph_name = args.graph_name
    write_to_csv=args.write_to_csv
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    rewrites_eval=load_eval(EVAL_PATH,REWRITE_METHODS)
    label_dict = create_label_dict(rewrites_eval, metric)
    sep_token="#" if col=="or_quac" else "_"
    corr_type=args.corr_type
    out_dir="{}/{}/{}/analysis/".format(qpp_res_dir_base,res_dir,col)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    features_turns_corrs = {}
    ret_res={rewrite_method:[] for rewrite_method in REWRITE_METHODS}
    fig, axs = plt.subplots(2, 2,sharex=False,sharey=True,figsize=(15,8))
    for j,rewrite_method in enumerate(REWRITE_METHODS):

        print("turn analysis for:",rewrite_method)
        method_label=label_dict[rewrite_method]
        print("num queries:",len(method_label))
        feature_corrs = {}
        #subgraph
        #non zero results
        #method_label={k:v  for k,v in method_label.items() if  v<1 }
        #print("num non one queries:", len(method_label))
        #eval_df=rewrites_eval[rewrite_method]
        #print(eval_df[eval_df.metric.str.startswith(metric)].groupby("value").count())
        all_qids=list(method_label.keys())
        all_labels = [method_label[qid] for qid in all_qids]
        print("num non zero:",len([x for x in all_labels if x>0]))
        print("num less than 0.1 :", len([x for x in all_labels if x < 0.1 and x>0]))
        print("num less than 0.05 :", len([x for x in all_labels if x < 0.05 and x>0]))

        features_turns_corrs[rewrite_method]={}
        qids={}
        for i in range(17):
            tid=str(i)
            turn_qids=[k for k,v in method_label.items() if k.split(sep_token)[1]==tid]

            if len(turn_qids)==0:
                continue
            qids[i]=turn_qids
            turn_labels=[method_label[qid] for qid in turn_qids]
            avg_metric_val=round(np.mean(turn_labels),3)
            print("turn ",tid,len(turn_labels),avg_metric_val)
            if len(turn_labels) >= min_turn_samples:
                ret_res[rewrite_method].append(avg_metric_val)


        for feature in features:
            print("calc feature:",feature)
            feature_turns_corrs=[]
            feature_val_path = "{}/{}/{}/cache/{}_{}.json".format(qpp_res_dir_base,res_dir,col,feature,metric)
            with open(feature_val_path) as f:
                feature_val=json.load(f)
            feature_val_lst=[feature_val[rewrite_method][qid] for qid in all_qids]
            all_kendal, _ = kendalltau(all_labels, feature_val_lst)
            all_corr,_=pearsonr(all_labels,feature_val_lst)
            all_spear,_=spearmanr(all_labels,feature_val_lst)
            feature_corrs[feature]=calc_topic_corr(feature_val[rewrite_method], method_label, corr_type)
            print("person corr:",round(all_corr,3),"kendal corr:",round(all_kendal,3),"spearman corr:",round(all_spear,3))
            #acc,num_pairs=calc_topic_pairwise_acc(feature_val[rewrite_method],method_label,False)
            #print("acc:",acc,"num pairs:",num_pairs)
            acc, num_pairs = calc_topic_pairwise_acc(feature_val[rewrite_method], method_label)
            print("per turn acc:", acc, "per turn num pairs:", num_pairs)
            for i,turn_qids in qids.items():
                if len(turn_qids)<min_turn_samples:
                    continue
                #turn_labels = [method_label[qid] for qid in turn_qids]
                #turn_features = [feature_val[rewrite_method][qid] for qid in turn_qids]
                #turn_corr,p_val=kendalltau(turn_labels,turn_features)
                turn_labels = {qid:method_label[qid] for qid in turn_qids}
                turn_corr=calc_topic_corr(feature_val[rewrite_method],turn_labels,corr_type)
                feature_turns_corrs.append(turn_corr)
                print("turn",i,len(turn_qids),round(turn_corr,3))
            features_turns_corrs[feature]=feature_turns_corrs

        cur_ax = axs.flat[j]
        cur_ax.set_title(rewrite_method, fontsize=16)
        cur_ax.label_outer()
        for feature in features:
            num_turns =len(features_turns_corrs[feature])
            cur_ax.plot(range(1,num_turns+1),features_turns_corrs[feature],'o-',
                        label=METHOD_DISPLAY_NAME.get(feature,feature.replace("_"," ")))
        cur_ax.legend()
        '''
        plt.figure(figsize=(18,10))
        for feature in features:
            num_turns =len(features_turns_corrs[feature])
            plt.plot(range(1,num_turns+1),features_turns_corrs[feature],'o-',label=feature.replace("_"," "))
        plt.title("qpp {} results for {} rewrite method({})".format(corr_type,rewrite_method,col.replace("_"," ")))
        plt.xlabel("turn number")
        plt.ylabel(corr_type)
        plt.legend()
        file_name="{}/{}_{}_{}_{}.png".format(out_dir,graph_name,rewrite_method,corr_type,metric)
        plt.savefig(file_name)
        plt.close()
        '''
        method_turn_res=[]
        if len(features)>0 and write_to_csv:
            num_turns = len(features_turns_corrs[features[0]])
            for i in range(num_turns):
                row = {"turn number": i + 1, "num queries": len(qids[i if col == "or_quac" else i + 1])}
                for feature in features:
                    row[feature.replace("_", " ")] = round(features_turns_corrs[feature][i], 3)
                method_turn_res.append(row)
            all_turns_row = {"turn number": "all turns", "num queries": len(all_qids)}
            for feature in features:
                all_turns_row[feature.replace("_", " ")] = round(feature_corrs[feature], 3)
            method_turn_res.append(all_turns_row)
            method_turn_res_df = pd.DataFrame(method_turn_res)
            method_turn_res_df_path = "{}/{}_{}_{}_{}.csv".format(out_dir, graph_name, rewrite_method, corr_type,
                                                                  metric)
            method_turn_res_df.to_csv(method_turn_res_df_path, index=False)
            print(method_turn_res_df)

    fig.supxlabel("turn number", fontsize=16)
    fig.supylabel(corr_type, fontsize=16)
    file_name = "{}/{}_{}_{}.png".format(out_dir, graph_name, corr_type, metric)
    plt.savefig(file_name,dpi=180)
    plt.close()
    create_ret_turn_graph(ret_res,out_dir,metric)








