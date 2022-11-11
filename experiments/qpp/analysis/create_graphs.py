import argparse

import matplotlib.pyplot as plt
import numpy as np
import spacy
from ..qpp_utils import create_label_dict,load_eval,load_data

DEFAULT_COL='or_quac'
DEFAULT_RES_DIR="rerank_kld_100"
DEFAULT_VALID_RES_DIR="rerank_valid_kld_100"
DEFAULT_TRAIN_RES_DIR="rerank_train_kld_100"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
DEFAULT_QUERY_FIELD='second_stage_queries'

def create_qury_avg_length_size_by_turn(col,res_dir,query_rewrite_field,output_path):
    eval_path = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    runs_path = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
    runs, rewrites, rewrites_eval, turns_text = load_data(["all","quretec"], eval_path, runs_path, query_rewrite_field,
                                                          col)
    english = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser'])
    split_token = "#" if col == 'or_quac' else '_'
    tids_lengths={}
    num_turns = max([int(k.split(split_token)[1]) for k in rewrites['all'].keys()])
    tids_labels=[x for x in range(1,num_turns+1)]
    tid_align=-1 if col=="or_quac" else 0
    x = np.arange(len(tids_labels))  # the label locations
    width = 0.35  # the width of

    fig, ax = plt.subplots(figsize=(15, 6))
    bar_pos = - width / 2
    for method,method_rewrites in rewrites.items():
        print(method)
        for qid,query in method_rewrites.items():
            print(qid,query)
            q_length=len([t.lemma_ for t in english(query) ])
            sid, tid = qid.split(split_token)
            tids_lengths[tid] = tids_lengths.get(tid, []) + [q_length]
        res={tid:round(np.mean(v),1) for tid,v in tids_lengths.items()}
        res_values=[res[str(tid + tid_align)] for tid in tids_labels]
        recs=ax.bar(x +bar_pos, res_values, width, label=method)
        ax.bar_label(recs, padding=3)
        bar_pos+=width
    ax.set_ylabel('average query length')
    ax.set_title('{} average query length by turn'.format(col.replace("_"," ")))
    ax.set_xticks(x, tids_labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_train_size_graph(col,res_dir,valid_res_dir,train_res_dir,output_path):
    test_data=calc_data_size_per_turn(col,res_dir)
    valid_data=calc_data_size_per_turn(col,valid_res_dir)
    train_data = calc_data_size_per_turn(col,train_res_dir)
    num_turns=len(test_data.keys())
    tids=[x for x in range(1,num_turns+1)]
    tid_align=-1 if col=="or_quac" else 0
    print("tids",tids,tid_align)
    test_values=[test_data[str(tid+tid_align)] for tid in tids ]
    valid_values = [valid_data[str(tid + tid_align)] for tid in tids]
    train_values = [train_data[str(tid + tid_align)] for tid in tids]
    x = np.arange(len(tids))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(15,6))
    rects1 = ax.bar(x - width, train_values, width, label='train')
    rects2 = ax.bar(x,valid_values, width, label='valid')
    rects3 = ax.bar(x + width, test_values, width, label='test')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('number of queries')
    ax.set_title('{} dataset size by turn'.format(col.replace("_"," ")))
    ax.set_xticks(x, tids)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

def calc_data_size_per_turn(col,res_dir):
    eval_path = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
    rewrites_eval=load_eval(eval_path,["t5"])
    label_dict = create_label_dict(rewrites_eval, "recip_rank")["t5"]
    tids={}
    split_token="#" if col=='or_quac' else '_'
    for qid in label_dict.keys():
        sid,tid=qid.split(split_token)
        tids[tid]=tids.get(tid,0)+1
    return tids





if __name__=="__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--metric", default="recip_rank")
    parser.add_argument("--col", default=DEFAULT_COL)
    parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
    parser.add_argument("--valid_res_dir", default=DEFAULT_VALID_RES_DIR)
    parser.add_argument("--train_res_dir", default=DEFAULT_TRAIN_RES_DIR)
    parser.add_argument("--qpp_res_dir_base",default=DEFAULT_QPP_RES_DIR)
    parser.add_argument("--query_rewrite_field",default=DEFAULT_QUERY_FIELD)
    args = parser.parse_args()
    col=args.col
    res_dir=args.res_dir
    valid_res_dir=args.valid_res_dir
    train_res_dir=args.train_res_dir
    qpp_res_dir_base=args.qpp_res_dir_base
    query_rewrite_field=args.query_rewrite_field
    #output_file="{}/{}/{}/analysis/dataset_size.png".format(qpp_res_dir_base,res_dir,col)
    #create_train_size_graph(col, res_dir,valid_res_dir,train_res_dir, output_file)
    output_file="{}/{}/{}/analysis/average_query_length.png".format(qpp_res_dir_base,res_dir,col)
    create_qury_avg_length_size_by_turn(col,res_dir,query_rewrite_field,output_file)

