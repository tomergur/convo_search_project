from transformers import AutoTokenizer
import json
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

def serialize_example(input_ids, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids["input_ids"])),
        "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids["attention_mask"])),
        "token_type_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids["token_type_ids"])),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    return example.SerializeToString()


def serialize_query_passage_pair(qid, pid, query, passage, label, tokenizer, doc2q, writer):
    num_queries = np.random.choice(POSSIBLE_NUM_QUERIES)

    expanded_doc = "[SEP]".join([passage] + doc2q[pid][:num_queries]) if not ONLY_QUERIES else "[SEP]".join(
        doc2q[pid][:num_queries])
    tokenized_row = tokenizer(query, expanded_doc, max_length=512, truncation=True, padding='max_length',
                              return_token_type_ids=True)
    writer.write(serialize_example(tokenized_row, label))


if __name__ == "__main__":
    # constants
    doc2q_file = "/v/tomergur/convo/ms_marco/ms_marco_doc2q_res.json"
    collection_file = "/v/tomergur/convo/ms_marco/collection.tsv"
    tokenizer_name = "castorini/monobert-large-msmarco-finetune-only"
    qrel_file = "/v/tomergur/convo/ms_marco/qrels.dev.tsv"
    queries_file = "/v/tomergur/convo/ms_marco/queries.train.tsv"

    #output_path_format = "/v/tomergur/convo/ms_marco/records_train_only_q/rec_{}.tfrecords"
    #output_path_format = "/v/tomergur/convo/ms_marco/records_dev_only_q/rec_{}.tfrecords"
    #output_path_format = "/v/tomergur/convo/ms_marco/records_dev_exp_doc_5/rec_{}.tfrecords"
    #output_path_format = "/v/tomergur/convo/ms_marco/records_train_exp_doc_5/rec_{}.tfrecords"
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True,
                        help='the name of the current run. will be the name of the output files')
    parser.add_argument('--max_rows',type=int,default=6400000)
    parser.add_argument('--max_per_file',type=int,default=29*1000)
    parser.add_argument('--output_path_format',required=True)
    parser.add_argument('--pair_mode',default=False,action='store_true')
    parser.add_argument('--only_queries',default=False,action='store_true')
    args=parser.parse_args()
    input_file = args.input_file
    output_path_format=args.output_path_format
    max_rows = args.max_rows
    max_rows_per_file =29*1000
    #max_rows = 6700000
    #max_rows_per_file = 58 * 1000
    IS_PAIR_MODE = args.pair_mode
    ONLY_QUERIES = args.only_queries
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # POSSIBLE_NUM_QUERIES = [5, 10, 20, 40]
    POSSIBLE_NUM_QUERIES = [5]

    if not IS_PAIR_MODE:
        qrel = pd.read_csv(qrel_file, header=None, names=["qid", "Q0", "pid", "grade"], dtype={"qid": str, "pid": str},
                           delimiter="\t")
        print(qrel)
        grade_col = qrel.set_index(["qid", "pid"]).grade
    else:
        collection = pd.read_csv(collection_file, header=None, names=["pid", "passage"], dtype={"pid": str},
                                 delimiter="\t")
        collection = collection.set_index("pid")
        queries = pd.read_csv(collection_file, header=None, names=["qid", "query"], dtype={"qid": str}, delimiter="\t")
        queries = queries.set_index("qid")
    with open(doc2q_file) as f:
        doc2q = json.load(f)
    writer=None
    file_idx=0
    with open(input_file) as f_inputs:
        for i, doc_row in tqdm.tqdm(enumerate(f_inputs), total=max_rows):
            if i >= max_rows:
                break
            if i%max_rows_per_file==0:
                writer=tf.io.TFRecordWriter(output_path_format.format(file_idx))
                file_idx+=1

            if IS_PAIR_MODE:
                qid, pos_pid, neg_pid = doc_row.rstrip().split("\t")
                query = queries["query"].get(qid)
                pos_passage = collection.passage.get(pos_pid)
                neg_passage = collection.passage.get(neg_pid)
                serialize_query_passage_pair(qid, pos_pid, query, pos_passage, 1, tokenizer, doc2q, writer)
                serialize_query_passage_pair(qid, neg_pid, query, neg_passage, 0, tokenizer, doc2q, writer)
            else:
                qid, pid, query, passage = doc_row.rstrip().split("\t")
                label = grade_col.get((qid, pid), 0)
                label = min(label, 1)
                serialize_query_passage_pair(qid, pid, query, passage, label, tokenizer, doc2q, writer)
