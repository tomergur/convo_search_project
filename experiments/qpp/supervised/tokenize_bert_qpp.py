import argparse
from transformers import AutoTokenizer
import json
import tensorflow as tf
import pandas as pd
from pyserini.search import SimpleSearcher


def serialize_example(features, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=features["input_ids"])),
        "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=features["attention_mask"])),
        "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=features["token_type_ids"])),
        "labels": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
    }))
    return example.SerializeToString()


def truncate_query(query, tokenizer, max_length=128):
    q_tokens = tokenizer(query)
    q_len = len(q_tokens['input_ids'])
    if max_length >= q_len:
        return query

    '''
    query_turns=query.split("[SEP]")
    for i in range(1,len(query_turns)):
        truncated_query="[SEP]".join(query_turns[i:])
        q_tokens = tokenizer(truncated_query)
        q_len = len(q_tokens['input_ids'])
        if max_length >= q_len:
            print("truncated q:",truncated_query)
            return truncated_query
    '''

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


def serialize_query_passage_pair(qid, query, passage, label, tokenizer, writer):
    tokenized_row = tokenizer(query, passage, max_length=512, truncation=True, padding='max_length',
                              return_token_type_ids=True)
    writer.write(serialize_example(tokenized_row, label))


def serialize_dataset_row(qid, query, passage, label, tokenizer, writer):
    serialize_query_passage_pair(qid, query, passage, label, tokenizer, writer)


tokenizer_name = "bert-base-uncased"
# tokenizer_name = "castorini/monobert-large-msmarco-finetune-only"
QREL_PATH = "/lv_local/home/tomergur/convo_search_project/data/or_quac/qrels.txt"
RES_PATH = "/v/tomergur/convo/res"
EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval"
INDEXES_DIR = "/v/tomergur/convo/indexes"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='the name of the current run. will be the name of the output files')
    parser.add_argument("--setting_name", default="kld_100")
    parser.add_argument("--metric", default="recip_rank")
    # parser.add_argument('--max_rows',type=int,default=6400000)
    parser.add_argument('--col', default="or_quac")
    parser.add_argument('--max_per_file', type=int, default=200000)
    parser.add_argument('--output_path_format', required=True)
    parser.add_argument("--is_rerank", action='store_true', default=False)
    parser.add_argument("--append_history", action='store_true', default=False)
    parser.add_argument("--append_prev_turns", action='store_true', default=False)
    parser.add_argument("--select_tid",default=None)

    # parser.add_argument('--qrel_path',default=QREL_PATH)
    args = parser.parse_args()
    run_name = args.run_name
    setting_name = args.setting_name
    metric = args.metric
    col = args.col
    output_path_format = args.output_path_format
    max_rows_per_file = args.max_per_file
    append_history = args.append_history
    append_prev_turns = args.append_prev_turns
    assert (not (append_history and append_prev_turns))
    select_tid = args.select_tid
    query_field_name = "second_stage_queries" if args.is_rerank else 'first_stage_rewrites'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    input_file = "{}/{}/{}/{}_queries.json".format(RES_PATH, setting_name, col, run_name)
    with open(input_file) as f:
        queries = json.load(f)

    split_token = "#" if col == "or_quac" else "_"
    first_tid = 0 if col == "or_quac" else 1

    run_file = "{}/{}/{}/{}_run.txt".format(RES_PATH, setting_name, col, run_name)
    runs = pd.read_csv(run_file, header=None, names=["qid", "Q0", "docid", "ranks", "score", "info"],
                       delimiter="\t", dtype={"docid": str})
    file_idx = 1
    eval_file = "{}/{}/{}/{}.txt".format(EVAL_PATH, setting_name, col, run_name)
    eval_res = pd.read_csv(eval_file, header=None, names=["metric", "qid", "value"], delimiter="\t")
    eval_res = eval_res[eval_res.metric.str.startswith(metric)]
    print("eval res", eval_res.head())
    metrics_values = eval_res.set_index("qid").value.to_dict()
    searcher = SimpleSearcher("{}/{}".format(INDEXES_DIR, col))
    # unique for bert qpp
    top_docs = runs[runs.ranks == 1].set_index("qid").docid.to_dict()
    j = 0
    for i, (qid, query_json) in enumerate(queries.items()):
        print(i, qid)
        raw_query = query_json[query_field_name][0]
        sid, tid = qid.split(split_token)
        if select_tid is not None and select_tid!=tid:
            continue
        if append_history or append_prev_turns:
            if int(tid) > first_tid:
                hist = []
                for i in range(first_tid, int(tid)):
                    cur_turn_qid = sid + split_token + str(i)
                    if cur_turn_qid in queries:
                        hist_query = queries[cur_turn_qid]["query"] if append_history else \
                        queries[cur_turn_qid][query_field_name][0]
                        hist.append(hist_query)
                # hist=history[last_turn_qid][query_field_name][0]
                raw_query = " [SEP] ".join(hist + [raw_query])
        query = truncate_query(raw_query, tokenizer)
        print(query)
        if qid not in metrics_values:
            continue
        if j % max_rows_per_file == 0:
            writer = tf.io.TFRecordWriter(output_path_format.format(file_idx))
            file_idx += 1
        j += 1
        top_doc_id = top_docs[qid]
        doc = searcher.doc(top_doc_id)
        label = metrics_values[qid]
        passage = json.loads(doc.raw())["contents"]
        # print(passage)
        serialize_dataset_row(qid, query, passage, label, tokenizer, writer)
