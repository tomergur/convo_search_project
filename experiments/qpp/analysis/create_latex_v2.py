import argparse
import json
import os
import time
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats

from ..qpp_utils import create_label_dict,load_eval,calc_topic_corr,calc_topic_pairwise_acc,calc_topic_turn_corr,evaluate_topic_predictor
import matplotlib.pyplot as plt

DEFAULT_COL='or_quac'
DEFAULT_RES_DIR="rerank_kld_100"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG","NQC","clarity","bert_qpp","bert_qpp_or_quac"]
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG_norm","NQC_norm","clarity_norm","bert_qpp","many_turns_bert_qpp","many_turns_bert_qpp_online","ref_hist_bert_qpp"]
DEFAULT_SELECTED_FEATURES=["WIG_norm","WIG_norm_pt","NQC_norm","NQC_norm_pt","clarity_norm","clarity_norm_pt","bert_qpp","bert_qpp_pt","ref_hist_bert_qpp","ref_hist_bert_qpp_pt"]
'''
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG_norm","NQC_norm",
                           "clarity_norm","bert_qpp_or_quac","bert_qpp_topiocqa","bert_qpp_hist_or_quac","bert_qpp_hist_topiocqa"]
'''

#,"many_turns_bert_qpp_online"

DEFAULT_SELECTED_FEATURES=["WIG_norm","WIG_norm_pt","NQC_norm","NQC_norm_pt","clarity_norm","clarity_norm_pt"]





DEFAULT_SELECTED_FEATURES=["bert_qpp_cls","ref_hist_bert_qpp_cls","many_turns_bert_qpp_cls"]
BASELINE_METHODS={"bert_qpp_cls":"*"}
DEFAULT_SELECTED_FEATURES=["bert_qpp","ref_hist_bert_qpp","many_turns_bert_qpp_tokens","ref_rewrites_bert_qpp","rewrites_bert_qpp","ref_rewrites_many_turns_bert_qpp_tokens"]
DEFAULT_SELECTED_FEATURES=["bert_qpp","bert_qpp_pt","ref_hist_bert_qpp_1kturns","ref_hist_bert_qpp_2kturns","ref_hist_bert_qpp_3kturns","ref_hist_bert_qpp","ref_hist_bert_qpp_skturns","many_turns_bert_qpp_tokens_1kturns","many_turns_bert_qpp_tokens_2kturns","many_turns_bert_qpp_tokens_3kturns","many_turns_bert_qpp_tokens","many_turns_bert_qpp_tokens_skturns"]
DEFAULT_SELECTED_FEATURES=["bert_qpp","ref_hist_bert_qpp","ref_hist_bert_qpp_skturns","many_turns_bert_qpp_tokens","many_turns_bert_qpp_tokens_skturns"]
DEFAULT_SELECTED_FEATURES=["bert_qpp","ref_hist_bert_qpp","ref_hist_bert_qpp_skturns","many_turns_bert_qpp_tokens","many_turns_bert_qpp_tokens_skturns","ref_rewrites_bert_qpp","ref_rewrites_bert_qpp_all_methods","ref_rewrites_bert_qpp_quretec_methods","ref_rewrites_bert_qpp_t5_methods","ref_rewrites_bert_qpp_hqe_methods","rewrites_bert_qpp"]
DEFAULT_SELECTED_FEATURES=["bert_qpp","ref_hist_agg_bert_qpp","many_turns_bert_qpp_tokens"]
BASELINE_METHODS={"bert_qpp":"0","ref_hist_agg_bert_qpp":"1"}
#,"ref_rewrites_agg_bert_qpp":'2'
REWRITE_METHODS=['t5','all','hqe','quretec']
DEFAULT_REWRITE_METHODS=['all','quretec']
#REWRITE_METHODS=['all']

TWO_DIGITS_METRICS = ["PA","TPA"]

QPP_EVAL_METRIC=["TPA","PA"]
QPP_EVAL_METRIC=["pearson"]
QPP_EVAL_METRIC=["pearson","kendall"]


QPP_EVAL_METRIC=["sturn_1_pearson","sturn_5_pearson","sturn_10_pearson"]
QPP_EVAL_METRIC=["sturn_0_kendall","sturn_4_kendall","sturn_8_kendall"]

#QPP_EVAL_METRIC=["sturn_1_kendall","sturn_5_kendall","sturn_9_kendall"]
#QPP_EVAL_METRIC=["TPA","turn_pearson","turn_kendall"]

#or quac
QPP_EVAL_METRIC=["sturn_0_kendall","sturn_1_kendall","sturn_2_kendall","sturn_4_kendall","sturn_6_kendall","sturn_8_kendall"]
METRICS_DISPLAY_NAME={"turn_pearson":"T$\\rho$","turn_kendall":"TK","sturn_0_pearson":"$T_{1}\\rho$",
                      "sturn_1_pearson":"$T_{1}\\rho$","sturn_4_pearson":"$T_{5}\\rho$","sturn_5_pearson":"$T_{5}\\rho$",
                      "sturn_9_pearson":"T_{10}$\\rho$","sturn_10_pearson":"T_{10}$\\rho$","sturn_0_kendall":"$T_{1}K$",
                      "sturn_1_kendall":"$T_{2}K$","sturn_2_kendall":"$T_{3}K$","sturn_4_kendall":"$T_{5}K$",
                      "sturn_6_kendall":"$T_{7}K$","sturn_8_kendall":"$T_{9}K$"}


#topiocqa
QPP_EVAL_METRIC=["sturn_1_kendall","sturn_2_kendall","sturn_3_kendall","sturn_5_kendall","sturn_7_kendall","sturn_9_kendall"]
METRICS_DISPLAY_NAME={"turn_pearson":"T$\\rho$","turn_kendall":"TK","sturn_0_pearson":"$T_{1}\\rho$",
                      "sturn_1_pearson":"$T_{1}\\rho$","sturn_4_pearson":"$T_{5}\\rho$","sturn_5_pearson":"$T_{5}\\rho$",
                      "sturn_9_pearson":"T_{10}$\\rho$","sturn_10_pearson":"T_{10}$\\rho$","sturn_1_kendall":"$T_{1}K$",
                      "sturn_2_kendall":"$T_{2}K$","sturn_3_kendall":"$T_{3}K$","sturn_5_kendall":"$T_{5}K$",
                      "sturn_7_kendall":"$T_{7}K$","sturn_9_kendall":"$T_{9}K$"}

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
                     "many_turns_bert_qpp":"dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens": "dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens_skturns": "dialogue groupwise QPP(vary history length)",
                     "many_turns_bert_qpp_online": "dialogue groupwise QPP - online inference",
                     "many_turns_bert_qpp_hist": "dialogue groupwise QPP+raw history",
                     "many_turns_bert_qpp_prev": "dialogue groupwise QPP+previous queries",
                     "seq_qpp":"dialogue LSTM QPP",
                     "rewrites_bert_qpp": "rewrites groupwise qpp",
                     "ref_rewrites_bert_qpp":"Bert QPP -rewrites REF RBO",
                     "ref_rewrites_bert_qpp_all_methods": "Bert QPP -all rewrite REF RBO",
                     "ref_rewrites_bert_qpp_t5_methods": "Bert QPP -t5 rewrite REF RBO",
                     "ref_rewrites_bert_qpp_quretec_methods": "Bert QPP -quretec rewrite REF RBO",
                     "ref_rewrites_bert_qpp_hqe_methods": "Bert QPP -hqe rewrite REF RBO",
                     "ref_hist_bert_qpp_cls":"Bert QPP -REF RBO","many_turns_bert_qpp_cls":"dialogue groupwise QPP",
                     "ref_hist_bert_qpp": "Bert QPP - history REF RBO",
                     "ref_hist_bert_qpp_skturns": "Bert QPP - history REF RBO(varying history size)",
                     "ref_hist_bert_qpp":"Bert QPP - history REF RBO","ref_hist_bert_qpp_pt":"Bert QPP - REF RBO, HP per turn "}

def is_oracle(method_name):
    return ('manual' in method_name) or ('oracle' in method_name)

def annotate_result(result, method_name, metric_name, col_res, t_col_res):
    if metric_name in TWO_DIGITS_METRICS:
        result = round(float(result), 2)
        results_str = '{:.2f}'.format(result).replace("0.", ".")
    else:
        result = round(float(result), 3)
        results_str = '{:.3f}'.format(result).replace("0.", ".")
    if is_oracle(method_name):
        return results_str
    results = [round(float(col_res[method][metric_name]), 2 if metric_name in TWO_DIGITS_METRICS else 3) for method in col_res if not is_oracle(method)]
    if result >= max(results):
        results_str = '\\textbf{' + results_str + '}'
    if metric_name not in t_col_res:
        return results_str
    sgni_str = t_col_res[metric_name].get(method_name, None)
    if sgni_str:
        results_str = '$' + results_str + '^{' + sgni_str + '}$'
    return results_str


def get_method_row(method_name, res_dict, columns, sub_columns,table_type, tt_res={}):
    res = METHOD_DISPLAY_NAME.get(method_name, method_name.replace("_"," ")) if table_type!='right' else ''
    for col in columns:
        res += '&'
        col_res = res_dict[col]
        t_col_res = tt_res.get(col,{})
        values = [annotate_result(col_res[method_name][sub_col], method_name, sub_col, col_res, t_col_res) for sub_col in sub_columns]
        values_str = '&'.join(values)
        res += values_str
    res += '\\\\ \\hline'
    # remove trailing &
    if table_type=='right':
        res=res[1:]
    return res

def calc_table_header(columns, sub_columns,table_type):
    res = '||m{12em} | ' if table_type!="right" else '|'
    collection_columns = ' '.join(['c'] * len(sub_columns))
    res += '|'.join([collection_columns] * len(columns))
    res += ' |'
    return res

def get_column_row(columns, sub_columns,table_type):
    METHOD_COLUMN_NAME="predictor"
    if len(sub_columns)==1:
        col_list=[METHOD_COLUMN_NAME]+columns if table_type!='right' else columns
        return "&".join(col_list)+"\\\\ \\hline"
    res = '\multirow{2}{4em}{' + METHOD_COLUMN_NAME + '} & ' if table_type!='right' else ''
    columns_num = len(sub_columns)
    multi_cols = ['\multicolumn{' + str(columns_num) + '}{|c|}{' + col + '}' for col in columns]
    res += ' & '.join(multi_cols) + '\\\\ \cline{2-' + str(1 + columns_num * len(columns)) + '}'
    return res

def get_table_columns_names(columns, sub_columns):
    res = '&'
    columns_names = '&'.join(sub_columns)
    res += '&'.join([columns_names] * len(columns))
    res += '\\\\ \\hline'
    return res

def result_to_latex(res_dict,output_path,t_test_res,table_type,table_headline):
    with open(output_path, 'w') as output:
        #print('\\begin{center}', file=output)
        column_names=list(res_dict.keys())
        row_names=list(res_dict[column_names[0]].keys())
        sub_columns_names=list(res_dict[column_names[0]][row_names[0]].keys())
        if table_type in ["normal","header"]:
            table_header = calc_table_header(column_names, sub_columns_names, table_type)
            print('\\begin{tabular}{' + table_header + '} \\hline', file=output)
            if table_headline is not None:
                num_col = 1 + len(sub_columns_names) * len(column_names)
                headline_str = "\multicolumn{" + str(num_col) + "}{|c|}{" + table_headline + "} \\\\ \\hline"
                print(headline_str, file=output)
        table_collections_row = get_column_row(column_names, sub_columns_names,table_type)
        print(table_collections_row, file=output)
        if len(sub_columns_names)>1:
            table_col_names_line = get_table_columns_names(column_names, sub_columns_names)
            print(table_col_names_line, file=output)
        for row_name in row_names:
            print(get_method_row(row_name, res_dict, column_names, sub_columns_names,table_type,t_test_res), file=output)
        if table_type in ["normal","table_end"]:
            print('\\end{tabular}', file=output)
        #print('\\end{center}', file=output)

def t_test_paired(new_method, baseline, alpha=.05):
    t, p_val = stats.ttest_rel(new_method, baseline, alternative="greater")
    return t > 0 and p_val < alpha



def get_ttest_vals(split_res, baseline_methods=BASELINE_METHODS):
    res = {}
    print(baseline_methods)
    for col_name in split_res.keys():
        col_res = split_res[col_name]
        col_ttest = {}
        metrics = list(col_res.values())[0].keys()
        print(metrics)
        for metric in metrics:
            metric_res = {}
            for subcol_name, subcol_res in col_res.items():
                '''
                if method_name in baseline_methods:
                    continue
                '''
                sgni_symbols = [method_key for method_name, method_key in \
                                baseline_methods.items() if t_test_paired(subcol_res[metric], col_res[method_name][metric])]
                if len(sgni_symbols) > 0:
                    metric_res[subcol_name] = ''.join(sgni_symbols)
            col_ttest[metric] = metric_res
        res[col_name] = col_ttest
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--corr_type",default="pearson")
    parser.add_argument("--min_turn_samples",type=int,default=0)
    parser.add_argument("--table_type",default="normal")
    parser.add_argument("--output_file_name",default="latex_res.txt")
    parser.add_argument("--rewrite_methods",nargs="+",default=DEFAULT_REWRITE_METHODS)
    parser.add_argument("--subsamples_size",type=int,default=50)
    parser.add_argument("--add_col_header",action='store_true',default=False)
    parser.add_argument("--qpp_eval_metric",default="kendall")
    args=parser.parse_args()
    metric=args.metric
    col=args.col
    res_dir=args.res_dir
    features=args.features
    min_turn_samples = args.min_turn_samples
    qpp_res_dir_base = args.qpp_res_dir_base
    table_type = args.table_type
    output_file_name=args.output_file_name
    subsamples_size=args.subsamples_size
    REWRITE_METHODS=args.rewrite_methods
    add_col_header=args.add_col_header
    qpp_eval_metric=args.qpp_eval_metric

    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)

    sep_token="#" if col=="or_quac" else "_"
    corr_type=args.corr_type
    out_dir="{}/{}/{}/analysis/".format(qpp_res_dir_base,res_dir,col)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    latex_res={}
    features_exp={}
    splits_res={}
    display_metric_name = "K" if qpp_eval_metric == "kendall" else "PA"
    for feature in features:
        feature_eval = {}
        print("calc feature:", feature)
        start_time = time.time()
        exp_path = "{}/{}/{}/exp_per_turn_{}_{}_{}_30_{}.json".format(qpp_res_dir_base, res_dir, col,
                                                                       qpp_eval_metric,feature, metric,subsamples_size)

        with open(exp_path) as f:
            exp_turns = json.load(f)
        features_exp[feature]=exp_turns

    for rewrite_method in REWRITE_METHODS:
        print("calc res for:",rewrite_method)
        latex_res[rewrite_method]={}
        splits_res[rewrite_method]={}

        for feature in features:
            feature_eval={}
            split_eval={}
            feature_splits_res=features_exp[feature][rewrite_method]
            max_turns=len(feature_splits_res[0])
            turns=[0,1,2,4,6,8]
            turns=range(max_turns)
            for i in turns:
                turn_kendall=[feature_res[i] for feature_res in feature_splits_res]
                turn_res=round(np.mean(turn_kendall),3)
                #print("num splits",len(turn_kendall))
                feature_eval["$T_{"+str(i+1)+"}"+display_metric_name+"$"]=turn_res
                split_eval["$T_{"+str(i+1)+"}"+display_metric_name+"$"]=turn_kendall
                #print("eval table",i+1, turn_res)
            #print("feature value calc:", time.time() - start_time)
            latex_res[rewrite_method][feature] = feature_eval
            splits_res[rewrite_method][feature]=split_eval
        method_latex_res=splits_res[rewrite_method]
        for feature1,feature2 in itertools.combinations(method_latex_res.keys(),2):
            feature1_eval=method_latex_res[feature1]
            feature2_eval = method_latex_res[feature2]
            feature1_win=len([1 for k in feature1_eval.keys() if feature1_eval[k]>feature2_eval[k]])
            feature1_tie = len([1 for k in feature1_eval.keys() if feature1_eval[k] == feature2_eval[k]])
            feature1_lose = len([1 for k in feature1_eval.keys() if feature1_eval[k] < feature2_eval[k]])
            print(feature1,feature2,feature1_win,feature1_tie,feature1_lose)

    output_file="{}/{}".format(out_dir,output_file_name)
    t_test_res=get_ttest_vals(splits_res)
    #print("t_test_res",t_test_res)
    col_names={'or_quac':'OR QUAC','topiocqa':'TopioCQA'}
    table_main_row=col_names.get(col) if add_col_header else None
    result_to_latex(latex_res,output_file,t_test_res,table_type,table_main_row)








