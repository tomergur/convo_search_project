import argparse
from transformers import AutoTokenizer
import json
import tensorflow as tf
import pandas as pd
from pyserini.search import SimpleSearcher
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
VALID_DATASET_MODES=["dialogue","many_docs","dialogue_limited"]

def create_dialogue_limited_dataset(data_to_tokenize,max_rows_per_file,split_token,max_hist_length=3):
    j = 0
    file_idx = 1
    print("limited dataset")
    for qid, (query, q_passages, label) in data_to_tokenize.items():
        if label is None:
            print("missing label")
            continue
        if j % 100 == 0:
            print("num serialized:", j)
        if j % max_rows_per_file == 0:
            writer = tf.io.TFRecordWriter(output_path_format.format(file_idx))
            file_idx += 1
        j += 1
        sid, tid = qid.split(split_token)
        queries=[]
        passages=[]
        for i in range(max(int(tid)-max_hist_length,0),int(tid)):
            cur_turn_qid="{}{}{}".format(sid,split_token,i)
            #print("cur_turn_qid", qid, cur_turn_qid)
            if cur_turn_qid not in data_to_tokenize:
                continue
            #print("add history qid",qid,cur_turn_qid)
            cur_query, cur_psgs, turn_label = data_to_tokenize[cur_turn_qid]
            queries+=[cur_query]*len(cur_psgs)
            passages+=cur_psgs
        queries+=[query]*len(q_passages)
        passages+=q_passages
        labels= [label] if isinstance(label, float) else label
        serialize_dataset_row(queries, passages, labels, tokenizer, writer)



def create_many_docs_dataset(data_to_tokenize,max_rows_per_file):
    j = 0
    file_idx = 1
    for qid, (query, passages, label) in data_to_tokenize.items():
        if label is None:
            print("missing label")
            continue
        print(qid)
        if j % 100 == 0:
            print("num serialized:", j)
        if j % max_rows_per_file == 0:
            writer = tf.io.TFRecordWriter(output_path_format.format(file_idx))
            file_idx += 1
        j += 1
        queries=[query]*len(passages)

        labels= [label] if isinstance(label, float) else label
        serialize_dataset_row(queries, passages, labels, tokenizer, writer)

def create_dialogue_dataset(data_to_tokenize,max_rows_per_file,split_token,selected_tid=None):
    j = 0
    file_idx = 1

    for qid, (query, q_passages, label) in data_to_tokenize.items():
        if label is None:
            print("missing label")
            continue
        sid, tid = qid.split(split_token)
        if (selected_tid is None) and int(tid) < max_turn.get(sid):
            continue
        if selected_tid and tid!=selected_tid:
            #print(tid,selected_tid)
            continue
        #print(qid)
        if j % 100 == 0:
            print("num serialized:", j)
        if j % max_rows_per_file == 0:
            writer = tf.io.TFRecordWriter(output_path_format.format(file_idx))
            file_idx += 1
        j += 1
        queries = []
        passages = []
        labels = []
        for i in range(first_tid, int(tid)):
            cur_turn_qid = sid + split_token + str(i)
            if cur_turn_qid in data_to_tokenize:
                cur_query, cur_psgs, turn_label = data_to_tokenize[cur_turn_qid]
                queries.append(cur_query)
                passages.append(cur_psgs[0])
                labels.append(turn_label)
        labels.append(label)
        if selected_tid:
            labels=[label]
        queries.append(query)
        passages.append(q_passages[0])
        serialize_dataset_row(queries, passages, labels, tokenizer, writer)


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
    parser.add_argument("--dataset_mode",default="dialogue")
    parser.add_argument("--selected_tid",default=None)
    parser.add_argument("--max_top_docs",type=int,default=1)
    parser.add_argument("--qrel_path",default=None)
    parser.add_argument("--max_tid",type=int,default=None)

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
    dataset_mode=args.dataset_mode
    assert(dataset_mode in VALID_DATASET_MODES)
    max_top_docs=args.max_top_docs
    selected_tid=args.selected_tid
    qrel_path=args.qrel_path
    max_tid=args.max_tid
    query_field_name = "second_stage_queries" if args.is_rerank else 'first_stage_rewrites'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    input_file = "{}/{}/{}/{}_queries.json".format(RES_PATH, setting_name, col, run_name)
    qrel=pd.read_csv(qrel_path, header=None, names=["qid", "Q0", "docid", "grade"],
                    delimiter="\t", dtype={"docid": str}) if qrel_path else None
    with open(input_file) as f:
        queries = json.load(f)

    split_token = "#" if col == "or_quac" else "_"
    first_tid = 0 if col == "or_quac" else 1

    run_file = "{}/{}/{}/{}_run.txt".format(RES_PATH, setting_name, col, run_name)
    runs = pd.read_csv(run_file, header=None, names=["qid", "Q0", "docid", "ranks", "score", "info"],
                       delimiter="\t", dtype={"docid": str})
    eval_file = "{}/{}/{}/{}.txt".format(EVAL_PATH, setting_name, col, run_name)
    eval_res = pd.read_csv(eval_file, header=None, names=["metric", "qid", "value"], delimiter="\t")
    eval_res = eval_res[eval_res.metric.str.startswith(metric)]
    print("eval res", eval_res.head())
    metrics_values = eval_res.set_index("qid").value.to_dict()
    print("num labeled",len(metrics_values),"num queries:",len(queries))
    searcher = SimpleSearcher("{}/{}".format(INDEXES_DIR, col))
    # unique for bert qpp
    #top_docs = runs[runs.ranks == 1].set_index("qid").docid.to_dict()
    top_docs_df=runs[runs.ranks <= max_top_docs]
    top_docs ={qid: q_top_docs.docid.to_list() for qid,q_top_docs in top_docs_df.groupby('qid')}
    data_to_tokenize={}
    max_turn={}
    for i, (qid, query_json) in enumerate(queries.items()):
        #print(i, qid)
        raw_query = query_json[query_field_name][0]
        sid, tid = qid.split(split_token)
        if max_tid is not None and int(tid)>=max_tid+first_tid:
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
        #print(qid,"rewritten query",query)
        if qid not in metrics_values:
            print("not serilazing qid:",qid)
            continue
        max_turn[sid] = max(int(tid), max_turn.get(sid, 0))
        top_doc_ids = top_docs[qid]
        docs = [searcher.doc(top_doc_id) for top_doc_id in top_doc_ids]
        if len(top_doc_ids)<max_top_docs:
            print("not enough docs :",len(top_doc_ids))
            continue
        label = metrics_values.get(qid,DEFAULT_LABEL)
        if qrel is not None:
            #print("qrel",qrel)
            q_qrel=qrel[qrel.qid==qid]
            #print("q qrel",q_qrel)
            q_labels=q_qrel.set_index('docid').grade.to_dict()
            #print("q labels",q_labels)
            label=[q_labels.get(top_doc_id,0) for top_doc_id in top_doc_ids]

        passages = [json.loads(doc.raw())["contents"] for doc in docs]
        data_to_tokenize[qid]=(query,passages,label)
    if dataset_mode=="many_docs":
        create_many_docs_dataset(data_to_tokenize,max_rows_per_file)
    elif dataset_mode=="dialogue":
        create_dialogue_dataset(data_to_tokenize,max_rows_per_file,split_token,selected_tid)
    else:
        create_dialogue_limited_dataset(data_to_tokenize,max_rows_per_file,split_token)

