import argparse
from transformers import AutoTokenizer
import json
import tensorflow as tf
import pandas as pd
from pyserini.search import SimpleSearcher
from  experiments.qpp.qpp_utils import load_data, create_label_dict, create_ctx
import itertools

def serialize_example(features, labels):
    input_ids=list(itertools.chain(*features["input_ids"]))
    #print(len(input_ids))
    attention_mask=list(itertools.chain(*features["attention_mask"]))
    token_type_ids=list(itertools.chain(*features["token_type_ids"]))
    example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
        "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=token_type_ids)),
        "labels": tf.train.Feature(float_list=tf.train.FloatList(value=labels)),
    }))
    return example.SerializeToString()


def truncate_query(query, tokenizer, max_length=128):
    q_tokens = tokenizer(query)
    q_len = len(q_tokens['input_ids'])
    if max_length >= q_len:
        return query
    query_terms = query.split(" ")
    for i in range(1, len(query_terms)):
        truncated_query = " ".join(query_terms[i:])
        q_tokens = tokenizer(truncated_query)
        q_len = len(q_tokens['input_ids'])
        if max_length >= q_len:
            print("truncated q:", truncated_query)
            return truncated_query
    assert (False)
    return ""


def serialize_query_passage_pair(queries, passages, labels, tokenizer, writer):
    tokenized_row = tokenizer(queries, passages, max_length=512, truncation=True, padding='max_length',
                              return_token_type_ids=True)

    writer.write(serialize_example(tokenized_row,labels))


def serialize_dataset_row(queries, passages, labels, tokenizer, writer):
    serialize_query_passage_pair(queries, passages, labels, tokenizer, writer)


tokenizer_name = "bert-base-uncased"
# tokenizer_name = "castorini/monobert-large-msmarco-finetune-only"
QREL_PATH = "/lv_local/home/tomergur/convo_search_project/data/or_quac/qrels.txt"
RES_PATH = "/v/tomergur/convo/res"
EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval"
INDEXES_DIR = "/v/tomergur/convo/indexes"
DEFAULT_LABEL=None
REWRITE_METHODS = ['t5', 'all', 'hqe', 'quretec']


def create_dataset(data_to_tokenize,top_docs,max_rows_per_file):
    j = 0
    file_idx = 1
    for qid, (queries,passages,labels) in data_to_tokenize.items():
        #print(qid)
        #print(queries)
        #print(passages)
        #print(labels)
        assert(len(labels)==4)
        assert(len(queries)==4)
        if j % 100 == 0:
            print("num serialized:", j)
        if j % max_rows_per_file == 0:
            writer = tf.io.TFRecordWriter(output_path_format.format(file_idx))
            file_idx += 1
        j += 1
        queries_lst=[]
        for query in queries:
            queries_lst+=[query]*top_docs
        serialize_dataset_row(queries_lst, passages, labels, tokenizer, writer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting_name", default="kld_100")
    parser.add_argument("--metric", default="recip_rank")
    # parser.add_argument('--max_rows',type=int,default=6400000)
    parser.add_argument('--col', default="or_quac")
    parser.add_argument('--max_per_file', type=int, default=200000)
    parser.add_argument('--output_path_format', required=True)
    parser.add_argument("--is_rerank", action='store_true', default=True)
    parser.add_argument("--append_history", action='store_true', default=False)
    parser.add_argument("--append_prev_turns", action='store_true', default=False)
    parser.add_argument("--top_docs",type=int,default=1)
    parser.add_argument("--max_turn",type=int,default=9)
    args = parser.parse_args()
    setting_name = args.setting_name
    metric = args.metric
    col = args.col
    output_path_format = args.output_path_format
    max_rows_per_file = args.max_per_file
    append_history = args.append_history
    append_prev_turns = args.append_prev_turns
    assert (not (append_history and append_prev_turns))
    top_docs=args.top_docs
    max_tid=args.max_turn
    query_field_name = "second_stage_queries" if args.is_rerank else 'first_stage_rewrites'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(setting_name, col)
    RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(setting_name, col)
    runs, rewrites, rewrites_eval, turns_text = load_data(REWRITE_METHODS, EVAL_PATH, RUNS_PATH, query_field_name,
                                                          col)
    label_dict = create_label_dict(rewrites_eval, metric)
    ctx = create_ctx(runs, rewrites, turns_text, col)
    split_token = "#" if col == "or_quac" else "_"
    first_tid = 0 if col == "or_quac" else 1
    searcher = SimpleSearcher("{}/{}".format(INDEXES_DIR, col))
    # unique for bert qpp
    data_to_tokenize={}
    max_turn={}
    qids=list(label_dict[REWRITE_METHODS[0]].keys())
    for i, qid in enumerate(qids):
        sid, tid = qid.split(split_token)
        if int(tid)>=max_tid+first_tid:
            continue
        #print(i, qid)
        queries=[]
        passages=[]
        labels=[]
        contains_query=[qid in ctx[rewrite_method] for rewrite_method in REWRITE_METHODS]
        if False in contains_query:
            print("skipping query",qid)
            continue
        for rewrite_method in REWRITE_METHODS:
            method_rewrites=rewrites[rewrite_method]
            raw_query = method_rewrites[qid]
            method_ctx=ctx[rewrite_method]
            method_label=label_dict[rewrite_method]
            if append_history or append_prev_turns:
                if int(tid) > first_tid:
                    hist = []
                    for i in range(first_tid, int(tid)):
                        cur_turn_qid = sid + split_token + str(i)
                        if cur_turn_qid in method_rewrites:
                            hist_query = method_rewrites[cur_turn_qid] if append_history else \
                                turns_text[cur_turn_qid]
                            hist.append(hist_query)
                    # hist=history[last_turn_qid][query_field_name][0]
                    raw_query = " [SEP] ".join(hist + [raw_query])
            query = truncate_query(raw_query, tokenizer)
            # print(qid,"rewritten query",query)
            max_turn[sid] = max(int(tid), max_turn.get(sid, 0))
            top_doc_ids = [x[0] for x in method_ctx[qid]['res_list'][:top_docs]]
            docs = [searcher.doc(top_doc_id) for top_doc_id in top_doc_ids]
            label = method_label[qid]
            passages+=[json.loads(doc.raw())["contents"] for doc in docs]
            queries.append(query)
            labels.append(label)
        if len(passages)<len(REWRITE_METHODS)*top_docs:
            print("not enough docs:",qid,len(passages))
            continue
        data_to_tokenize[qid]=(queries,passages,labels)

    create_dataset(data_to_tokenize,top_docs,max_rows_per_file)
