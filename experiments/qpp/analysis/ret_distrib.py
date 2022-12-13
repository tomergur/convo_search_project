import argparse
import json
import os
import numpy as np
import pandas as pd

from ..qpp_utils import create_label_dict,load_eval,calc_topic_corr,calc_topic_pairwise_acc
from scipy.stats import pearsonr,kendalltau,spearmanr
import matplotlib.pyplot as plt

DEFAULT_COL='or_quac'
DEFAULT_RES_DIR='rerank_kld_100'
REWRITE_METHODS=['t5','all','hqe','quretec']
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--write_to_csv",action="store_true",default=False)
    args=parser.parse_args()
    metric=args.metric
    col=args.col
    res_dir=args.res_dir
    qpp_res_dir_base = args.qpp_res_dir_base
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    split_token="#" if col=='or_quac' else "_"
    rewrites_eval=load_eval(EVAL_PATH,REWRITE_METHODS,split_token)
    label_dict = create_label_dict(rewrites_eval, metric)
    #####
    cache_path="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/{}/{}/cache/".format(res_dir,col)
    json_res={k:[] for k in ["t5", "all","hqe","quretec"]}
    turns_params=[1,2,3,None]
    turns_suffix=["_1kturns","_2kturns","_3kturns",""]
    for param,suffix in zip(turns_params,turns_suffix):
        model_path="{}/many_turns_bert_qpp_tokens{}.json".format(cache_path,suffix)
        with open(model_path) as f:
            model_json=json.load(f)
            for method,cached_values in model_json.items():
                modified_cached_values=[]
                for v in cached_values:
                    new_params=[("max_seq_length",param)]+v[0]
                    modified_cached_values.append((new_params,v[1]))
                json_res[method]+=modified_cached_values
    with open("{}/many_turns_bert_qpp_tokens_skturns.json".format(cache_path),'w') as f:
        json.dump(json_res,f)
    #####
    pd.set_option('display.max_rows', None)
    for rewrite_method,labels in label_dict.items():
        print("rewrite method:",rewrite_method)
        data=[]
        for qid,label in labels.items():
            data.append({"qid":qid,"tid":qid.split(split_token)[1],"label":label})
        res_df=pd.DataFrame(data)
        #print(res_df.groupby('label').count())
        for tid,tid_df in res_df.groupby('tid'):
            print("rewrite method:", rewrite_method,"tid:",tid)
            #print(tid_df.groupby('label').count())
            df_len=len(tid_df)
            num_1=len(tid_df[tid_df.label==1])
            print("num equal 1:",num_1,round(num_1/df_len,2))
            num_0=len(tid_df[tid_df.label == 0])
            print("num equal 0:",num_0,round(num_0/df_len,2))
            print("other:",df_len-num_1-num_0,round((df_len-num_1-num_0)/df_len,2))


