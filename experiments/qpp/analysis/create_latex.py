import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from ..qpp_utils import create_label_dict,load_eval,calc_topic_corr,calc_topic_pairwise_acc,calc_topic_turn_corr,evaluate_topic_predictor
from scipy.stats import pearsonr,kendalltau,spearmanr
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



DEFAULT_SELECTED_FEATURES=["bert_qpp","bert_qpp_pt","st_bert_qpp_pt","bert_qpp_oracle","bert_qpp_oracle_pt",
                           "st_bert_qpp_oracle_pt"]
#
DEFAULT_SELECTED_FEATURES=["bert_qpp_or_quac","ref_hist_bert_qpp_or_quac","many_turns_bert_qpp_tokens_or_quac",
                           "bert_qpp_topiocqa","ref_hist_bert_qpp_topiocqa","many_turns_bert_qpp_tokens_topiocqa"]
REWRITE_METHODS=['t5','all','hqe','quretec']

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
QPP_EVAL_METRIC=["sturn_1_kendall","sturn_2_kendall","sturn_3_kendall","sturn_4_kendall","sturn_5_kendall","sturn_6_kendall","sturn_7_kendall","sturn_8_kendall","sturn_9_kendall"]
METRICS_DISPLAY_NAME={"turn_pearson":"T$\\rho$","turn_kendall":"TK","sturn_0_pearson":"$T_{1}\\rho$",
                      "sturn_1_pearson":"$T_{1}\\rho$","sturn_4_pearson":"$T_{5}\\rho$","sturn_5_pearson":"$T_{5}\\rho$",
                      "sturn_9_pearson":"T_{10}$\\rho$","sturn_10_pearson":"T_{10}$\\rho$","sturn_1_kendall":"$T_{1}K$",
                      "sturn_2_kendall":"$T_{2}K$","sturn_3_kendall":"$T_{3}K$","sturn_4_kendall":"$T_{4}K$","sturn_5_kendall":"$T_{5}K$",
                      "sturn_6_kendall":"$T_{6}K$","sturn_7_kendall":"$T_{7}K$","sturn_8_kendall":"$T_{8}K$","sturn_9_kendall":"$T_{9}K$"}

METHOD_DISPLAY_NAME={"WIG_norm":"WIG","clarity_norm":"clarity","NQC_norm":"NQC","bert_qpp":"Bert QPP",
                     "WIG_norm_pt":"WIG -HP per turn","clarity_norm_pt":"clarity -HP per turn",
                     "NQC_norm_pt":"NQC -HP per turn","bert_qpp_pt":"Bert QPP -HP per turn",
                     "st_bert_qpp_pt":"Bert QPP - fine tuned and HP per turn",
                     "st_bert_qpp_oracle_pt":"BERT QPP - fine tuned and HP per turn(HP selected by oracle)",
                     "bert_qpp_oracle": "BERT QPP - (HP selected by oracle for all turns)",
                     "bert_qpp_cls":"Bert QPP(CE loss)","bert_qpp_reg":"Bert QPP(MSE loss)",
                     "bert_qpp_or_quac":"[Or QUAC]Bert QPP",
                     "bert_qpp_topiocqa":"[TopioCQA]Bert QPP",

                     "bert_qpp_hist":"Bert QPP+ raw history","bert_qpp_hist_or_quac":"Bert QPP+history fine-tuned on Or QUAC",
                     "bert_qpp_hist_topiocqa":"Bert QPP+history fine-tuned on TopioCQA",
                     "bert_qpp_prev":"Bert QPP+previous queries",
                     "many_turns_bert_qpp":"dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens": "dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens_or_quac": "[Or QUAC]dialogue groupwise QPP",
                     "many_turns_bert_qpp_tokens_topiocqa": "[TopioCQA]dialogue groupwise QPP",
                     "many_turns_bert_qpp_online": "dialogue groupwise QPP - online inference",
                     "many_turns_bert_qpp_hist": "dialogue groupwise QPP+raw history",
                     "many_turns_bert_qpp_prev": "dialogue groupwise QPP+previous queries",
                     "ref_hist_bert_qpp_or_quac":"[Or QUAC]Bert QPP -REF RBO","ref_hist_bert_qpp_topiocqa":"[TopioCQA]Bert QPP -REF RBO",
                     "ref_hist_bert_qpp":"Bert QPP -REF RBO","ref_hist_bert_qpp_pt":"Bert QPP - REF RBO, HP per turn "}

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
    if abs(result) >= max([abs(x) for x in results]):
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

def result_to_latex(res_dict,output_path,table_type):
    with open(output_path, 'w') as output:
        print('\\begin{center}', file=output)
        column_names=list(res_dict.keys())
        row_names=list(res_dict[column_names[0]].keys())
        sub_columns_names=list(res_dict[column_names[0]][row_names[0]].keys())
        table_header = calc_table_header(column_names, sub_columns_names,table_type)
        print('\\begin{tabular}{' + table_header + '} \\hline', file=output)
        table_collections_row = get_column_row(column_names, sub_columns_names,table_type)
        print(table_collections_row, file=output)
        if len(sub_columns_names)>1:
            table_col_names_line = get_table_columns_names(column_names, sub_columns_names)
            print(table_col_names_line, file=output)
        for row_name in row_names:
            print(get_method_row(row_name, res_dict, column_names, sub_columns_names,table_type,{}), file=output)
        print('\\end{tabular}', file=output)
        print('\\end{center}', file=output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
    parser.add_argument("--rewrite_methods", nargs='+', default=['all', 'quretec'])
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--corr_type",default="pearson")
    parser.add_argument("--min_turn_samples",type=int,default=0)
    parser.add_argument("--table_type",default="normal")
    parser.add_argument("--output_file_name",default="latex_res.txt")
    args=parser.parse_args()
    metric=args.metric
    col=args.col
    res_dir=args.res_dir
    features=args.features
    min_turn_samples = args.min_turn_samples
    qpp_res_dir_base = args.qpp_res_dir_base
    table_type = args.table_type
    output_file_name=args.output_file_name
    REWRITE_METHODS = args.rewrite_methods
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    sep_token="#" if col=="or_quac" else "_"
    rewrites_eval=load_eval(EVAL_PATH,REWRITE_METHODS,sep_token)
    label_dict = create_label_dict(rewrites_eval, metric)

    corr_type=args.corr_type
    out_dir="{}/{}/{}/analysis/".format(qpp_res_dir_base,res_dir,col)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    latex_res={}
    for rewrite_method in REWRITE_METHODS:
        print("calc res for:",rewrite_method)
        method_label = label_dict[rewrite_method]
        latex_res[rewrite_method]={}
        for feature in features:
            feature_eval={}
            print("calc feature:", feature)
            if "cast" not in col or True:
                feature_val_path = "{}/{}/{}/cache/{}_{}.json".format(qpp_res_dir_base, res_dir, col, feature, "recip_rank")
                with open(feature_val_path) as f:
                    feature_val = json.load(f)
            start_time=time.time()
            for eval_metric in QPP_EVAL_METRIC:
                display_name=METRICS_DISPLAY_NAME.get(eval_metric,eval_metric)
                if "cast" in col and False:
                    table_path="{}/{}/{}/{}_{}_{}_30.csv".format(qpp_res_dir_base, res_dir, col,eval_metric,feature,metric)
                    qpp_eval_table=pd.read_csv(table_path)
                    print("eval table",qpp_eval_table.at[0,rewrite_method])
                    feature_eval[display_name] =qpp_eval_table.at[0,rewrite_method]
                else:
                    feature_eval[display_name]=evaluate_topic_predictor(feature_val[rewrite_method],method_label,eval_metric)
            print("feature value calc:", time.time() - start_time)
            latex_res[rewrite_method][feature] = feature_eval
    output_file="{}/{}".format(out_dir,output_file_name)
    result_to_latex(latex_res,output_file,table_type)








