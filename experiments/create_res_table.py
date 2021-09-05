# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:26:30 2019

@author: tomer
"""
import os
import csv
import scipy.stats as stats
import pandas as pd

EVAL_PATH = "./data/eval/"
DEF_OUTPUT = './data/result_table.txt'

METHOD_COLUMN_NAME = 'qid'
TT_INPUT_PATH = 'C:/Users/tomer/Documents/Python Scripts/retrival/data/q_summary'
INPUT_PATH = 'C:/Users/tomer/Documents/Python Scripts/retrival/data/q_summary'
BASELINE_METHODS = {'raw': '0', 'all': '1', 't5': '2'}
BASELINE_METHODS_BASE = {k + "_base": v for k, v in BASELINE_METHODS.items()}
SELECTED_BASELINE = BASELINE_METHODS

METHOD_NAMES = ['raw', 'all', 'quretec', 't5', 't5_fuse2', 't5_fuse3', 't5_fuse4', 't5_fuse5','t5_prev','t5_prev2','t5_prev3', 'manual']
METHOD_NAMES = ['raw', 'all', 'quretec', 't5', 'manual']
BASE_NAMES = [x + "_base" for x in METHOD_NAMES]
SELECTED_NAMES = METHOD_NAMES

METHOD_DISPLAY_NAMES = {'raw': 'current turn', 'all': 'all previous turns', 'quretec': 'QuReTeC', 't5': 'T5',
                        't5_fuse2': 'T5(fuse 2)', 't5_fuse3': 'T5(fuse 3)', 't5_fuse4': 'T5(fuse 4)',
                        't5_fuse5': 'T5(fuse 5)', 't5_prev': 'T5(window 1)', 't5_prev2': 'T5(window 2)',
                        't5_prev3': 'T5(window 3)'}


def t_test_paired(new_method, baseline, alpha=.05):
    t, p_val = stats.ttest_rel(new_method, baseline)
    return t > 0 and p_val < alpha


def get_all_methods(col_dir):
    res = {}
    for file_name in os.listdir(col_dir):
        res[file_name[:-3]] = pd.read_csv(col_dir + '/' + file_name)


def is_sgni(method_df, baseline_df, metric):
    method_metric_dict = pd.Series(method_df[metric].values, index=method_df.qid).to_dict()
    baseline_metric_dict = pd.Series(baseline_df[metric].values, index=baseline_df.qid).to_dict()
    qids = list(method_metric_dict.keys())
    method_list = [method_metric_dict[qid] for qid in qids]
    baseline_list = [baseline_metric_dict[qid] for qid in qids]
    return t_test_paired(method_list, baseline_list)


def get_ttest_vals(res_dfs, metrics=['map', 'p@5', 'ndcg@20'], baseline_methods=BASELINE_METHODS):
    res = {}
    print(baseline_methods)

    for col_name in res_dfs.keys():
        col_res = res_dfs[col_name]
        # print(col_res)
        col_ttest = {}
        for metric in metrics:
            metric_res = {}
            for method_name, method_df in col_res.items():
                '''
                if method_name in baseline_methods:
                    continue
                '''
                sgni_symbols = [method_key for method_name, method_key in \
                                baseline_methods.items() if is_sgni(method_df, col_res[method_name], metric)]
                if len(sgni_symbols) > 0:
                    metric_res[method_name] = ''.join(sgni_symbols)
            col_ttest[metric] = metric_res
        res[col_name] = col_ttest
    return res


def calc_table_header(collections, columns):
    res = '||l | '
    collection_columns = ' '.join(['c'] * len(columns))
    res += '|'.join([collection_columns] * len(collections))
    res += ' |'
    return res


def get_collection_row(collections, columns):
    res = '\multirow{2}{4em}{' + METHOD_COLUMN_NAME + '} & '
    columns_num = len(columns)
    multi_cols = ['\multicolumn{' + str(columns_num) + '}{|c|}{' + col + '}' for col in collections]
    res += ' & '.join(multi_cols) + '\\\\ \cline{2-' + str(1 + columns_num * len(collections)) + '}'
    return res


def get_table_columns_names(collections, columns):
    res = '&'
    columns_names = '&'.join(columns)
    res += '&'.join([columns_names] * len(collections))
    res += '\\\\ \\hline'
    return res


def is_oracle(method_name):
    return 'manual' in method_name


'''
def is_oracle(name):
    return False
'''


def annotate_result(result, method_name, metric_name, col_res, t_col_res):
    result = round(float(result), 3)
    results_str = '{:.3f}'.format(result).replace("0.", ".")
    if is_oracle(method_name):
        return results_str
    results = [round(float(col_res[method][metric_name]), 3) for method in col_res if not is_oracle(method)]
    if result >= max(results):
        results_str = '\\textbf{' + results_str + '}'
    if metric_name not in t_col_res:
        return results_str
    sgni_str = t_col_res[metric_name].get(method_name, None)
    if sgni_str:
        results_str = '$' + results_str + '^{' + sgni_str + '}$'
    return results_str


def get_method_row(method_name, res_dict, collections, columns, tt_res):
    res = method_name.replace("_base", "").replace("_rerank", "")
    res = METHOD_DISPLAY_NAMES.get(res, res)
    for collection in collections:
        res += '&'
        col_res = res_dict[collection]
        t_col_res = tt_res[collection]
        values = [annotate_result(col_res[method_name][col], method_name, col, col_res, t_col_res) for col in columns]
        values_str = '&'.join(values)
        res += values_str
    res += '\\\\ \\hline'
    return res


def result_to_latex(res_dict, method_names, output_path, tt_res):
    collections = list(res_dict.keys())
    columns = list(res_dict[collections[0]][method_names[-1]].keys())
    del columns[columns.index(METHOD_COLUMN_NAME)]
    print(columns)
    # del columns[columns.index('map_all')]
    with open(output_path, 'w') as output:
        print('\\begin{center}', file=output)
        table_header = calc_table_header(collections, columns)
        print('\\begin{tabular}{' + table_header + '} \\hline', file=output)
        table_collections_row = get_collection_row(collections, columns)
        print(table_collections_row, file=output)
        table_col_names_line = get_table_columns_names(collections, columns)
        print(table_col_names_line, file=output)
        for method in method_names:
            print(get_method_row(method, res_dict, collections, columns, tt_res), file=output)
        print('\\end{tabular}', file=output)
        print('\\end{center}', file=output)


METRIC_DISPLAY_NAMES = {'recall_1000': 'recall@1000', 'map_cut_1000': "map@1000", 'ndcg_cut_3': "ndcg@3"}


def parse_results(res_path, selected_method):
    res = {}
    q_res = {}
    for subdir, dirs, files in os.walk(res_path):
        methods_col_res = {}
        methods_col_q_res = {}
        for file in files:
            method = file[:-4]
            if method not in selected_method:
                continue
            run_res = pd.read_csv(subdir + '/' + file, header=None, names=["metric", "qid", "val"], delimiter='\t')
            run_res.metric = run_res.metric.map(lambda x: x.strip())
            q_rows = []
            for qid, q_group in run_res.groupby("qid"):
                q_dict = q_group[["metric", "val"]].set_index("metric").to_dict()['val']
                q_dict['qid'] = qid
                q_rows.append(q_dict)
            run_res = pd.DataFrame(q_rows)
            run_res = run_res.rename(METRIC_DISPLAY_NAMES, axis=1)
            q_metrics = run_res[run_res.qid != "all"]
            all_metrics = run_res[run_res.qid == "all"]
            methods_col_res[method] = all_metrics
            methods_col_q_res[method] = q_metrics
        if len(methods_col_res) > 0:
            col_name = os.path.basename(subdir)
            res[col_name] = methods_col_res
            q_res[col_name] = methods_col_q_res
    return res, q_res


if __name__ == "__main__":
    res, q_res = parse_results(EVAL_PATH, SELECTED_NAMES)
    tt_res = get_ttest_vals(q_res, list(METRIC_DISPLAY_NAMES.values()), baseline_methods=SELECTED_BASELINE)
    result_to_latex(res, SELECTED_NAMES, DEF_OUTPUT, tt_res)
