import pytrec_eval
import numpy as np
import math
import json
import os
import pandas as pd

TARGET_COLUMNS = ["qid", "sid", "docid", "rank", "grade","collection"]

def calc_metric(run, metric='ndcg_cut_3'):
    qrel = {}
    scores = {}
    for qid, q_run in run.groupby('qid'):
        qrel[qid] = q_run.set_index('docid')['grade'].to_dict()
        scores[qid] = q_run.set_index('docid')['score'].to_dict()
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'map_cut_1000', 'ndcg_cut_3'})
    metrics = evaluator.evaluate(scores)
    # print(json.dumps(evaluator.evaluate(scores), indent=1))
    if 'all' in metrics:
        print("all in metrics")
    return np.mean([q[metric] for q in metrics.values()])


def output_run_file(output_path, runs):
    print("write output", output_path)
    with open(output_path, "w") as f:
        for qid, q_group in runs.groupby('qid'):
            for rank, doc in q_group.sort_values('score', ascending=False).reset_index().iterrows():
                docno = doc["docid"]
                score = doc["score"]
                f.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, docno, rank + 1, score, "convo"))

def create_folds(sids, num_split=5, valid_size=0.15):
    perm_sessions = np.random.permutation(sids).tolist()
    fold_size = math.floor(len(sids) / num_split)
    print(fold_size)
    num_test_session = num_split * fold_size
    test_folds = [perm_sessions[x:x + fold_size] for x in range(0, num_test_session, fold_size)]
    remaining_data=len(sids) -num_test_session
    print("num_test_sessions:",num_test_session,remaining_data)
    for i in range(remaining_data):
        test_folds[i].append(perm_sessions[num_test_session+i])
    all_train_sids = [np.random.permutation([sid for sid in sids if sid not in fold]).tolist() for fold in test_folds]
    valid_sessions = round(len(sids) * valid_size)
    print(valid_sessions)
    valid_folds = [fold[:valid_sessions] for fold in all_train_sids]
    train_folds = [fold[valid_sessions:] for fold in all_train_sids]
    return [{'train': d[0], 'valid': d[1], 'test': d[2]} for d in zip(train_folds, valid_folds, test_folds)]


def _load_dataset(input_dir,collection,dataset_name):
    dataset_path = "{}/cast{}/{}.csv".format(input_dir,collection,dataset_name)
    dataset=pd.read_csv(dataset_path)
    return dataset.assign(collection=[collection]*len(dataset))

def create_dataset(data_args):
    if data_args.collection=="both":
        dataset19=_load_dataset(data_args.input_dir,'19',data_args.dataset_name)
        dataset20=_load_dataset(data_args.input_dir,'20',data_args.dataset_name)
        dataset=dataset19.append(dataset20)
    else:
        dataset = _load_dataset(data_args.input_dir,data_args.collection,data_args.dataset_name)
    if len(data_args.select_features) > 0:
        dataset = dataset[TARGET_COLUMNS + data_args.select_features]
    return dataset

def load_or_create_folds(args,sids):
    if args.collection=="both":
        folds19=_load_or_create_folds_for_collection(args.input_dir,"19", args.folds_file_name,args.folds_num,sids)
        folds20 = _load_or_create_folds_for_collection(args.input_dir,"20", args.folds_file_name,args.folds_num,sids)
        folds=[]
        for f1,f2 in zip(folds19,folds20):
            folds.append({k:f1[k]+f2[k] for k in ["train","valid","test"]})
        return folds
    return _load_or_create_folds_for_collection(args.input_dir, args.collection, args.folds_file_name, args.folds_num,sids)

def _load_or_create_folds_for_collection(input_dir,collection,folds_file_name,folds_num,sids):
    folds_path = "{}/cast{}/{}".format(input_dir,collection,folds_file_name)
    if os.path.isfile(folds_path):
        with open(folds_path) as f:
            folds = json.load(f)
    else:
        sessions = sids
        folds = create_folds(sessions,folds_num )
        with open(folds_path, 'w') as f:
            json.dump(folds, f, indent=True)
    return folds

def write_outputs(args,run_name,res):
    for collection,res_col in res.groupby('collection'):
        output_path = "{}/cast{}/{}_model_run.txt".format(args.output_path, collection, run_name)
        output_run_file(output_path, res_col)