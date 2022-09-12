import argparse
import json
import pandas as pd
import itertools
from .qpp_utils import load_data,create_label_dict
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
REWRITE_METHODS=['t5','all','hqe','quretec']
def calc_metric(scores):
    res=[]
    for qid,q_rows in scores.groupby(['qid']):
        sorted_rows=q_rows.sort_values('score',ascending=False)
        #print(sorted_rows,sorted_rows.grade.tolist(),sorted_rows.grade.tolist()[0])
        res.append(sorted_rows.grade.tolist()[0])
    return round(np.mean(res),3)

DEFAULT_RES_DIR="kld"
DEFAULT_COL = "cast19"
DEFAULT_QPP_RES_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/rewrites_comp/"
res_file_name="pre_ret_qpp.json"
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG","NQC","clarity",
                           "turn_jaccard","is_t5","is_all","is_hqe","is_quretec"]
DEFAULT_SELECTED_FEATURES=["q_len","max_idf","avg_idf","max_scq","avg_scq","max_var","avg_var","WIG_norm","NQC_norm","clarity_norm",
                           "turn_jaccard","is_t5","is_all","is_hqe","is_quretec"]
DEFAULT_C = [1e-3, 1e-2, .1]

def run_infrence(dataset, model, normalizer):
    x, labels = dataset
    x_norm = normalizer.transform(x)
    scores = model.decision_function(x_norm)
    print("scores_shape", scores.shape, "x shape", x_norm.shape)
    # print("scores",scores)
    labels = labels.assign(score=scores)
    #print(model.coef_, x.columns)
    return labels

def train_model(train_dataset, c_val, penalty="l2"):
    start_time = time.time()
    x_pair, y = train_dataset
    print("train model with cval:", c_val)
    model = LinearSVC(C=c_val, dual=False, penalty=penalty)
    #model=SVC(C=c_val,kernel="poly",degree=2)
    model.fit(x_pair, y)
    print("finish training model. time is:{}".format(time.time() - start_time))
    return model

def create_pairwise_data(x, target_df, label_column='grade'):
    x_res = []
    y_res = []
    k = 0
    assert (len(target_df) == x.shape[0])
    target_df = target_df.reset_index(drop=True)
    for qid, q_group in target_df.groupby('qid'):
        # print(qid)
        # q_time=time.time()
        label_col = target_df[label_column].tolist()
        for i, j in itertools.combinations(q_group.index.values.tolist(), 2):
            # print(i,j)
            target_i = label_col[i]
            target_j = label_col[j]
            if target_i == target_j:
                continue
            new_x = x[i, :] - x[j, :]
            new_y = np.sign(target_i - target_j)
            if new_y != (-1) ** k:
                new_y = - new_y
                new_x = -new_x
            k += 1
            x_res.append(new_x)
            y_res.append(new_y)
            # print(new_x.shape)
        # print("pair q time:",time.time()-q_time)

    return np.array(x_res), np.array(y_res)

def build_dataset(features_values,label_dict):
    TARGET_COLUMNS = ["qid", "grade","method"]
    method_names=list(label_dict.keys())
    labeled_qids=list(label_dict[method_names[0]].keys())
    feature_dicts={k:v.set_index("qid").to_dict() for k,v in features_values.items()}
    feature_names=list(features_values.keys())
    feature_qids=list(feature_dicts[feature_names[0]][method_names[0]].keys())
    qids=[qid for qid in labeled_qids if qid in feature_qids]
    print(qids)
    rows=[]

    #print(feature_dicts)
    for qid in qids:
        for method_name in method_names:
            q_row={"qid":qid,"grade":label_dict[method_name][qid],"method":method_name}
            for feature,feature_dict in feature_dicts.items():
                #print(feature_dict)
                q_row[feature]=feature_dict[method_name][qid]
            rows.append(q_row)
    dataset=pd.DataFrame(rows)
    #print(dataset)
    return  dataset.drop(TARGET_COLUMNS, axis=1), dataset[TARGET_COLUMNS]






def load_datasets(features,labels_dict,qpp_res_dir,fold_num):
    train_features,valid_features,test_features={},{},{}
    for feature in features:
        train_scores_path="{}/{}/train_fold_{}_scores.csv".format(qpp_res_dir,feature,fold_num)
        train_features[feature]=pd.read_csv(train_scores_path)
        valid_scores_path="{}/{}/valid_fold_{}_scores.csv".format(qpp_res_dir,feature,fold_num)
        valid_features[feature]=pd.read_csv(valid_scores_path)
        test_scores_path="{}/{}/test_fold_{}_scores.csv".format(qpp_res_dir,feature,fold_num)
        test_features[feature]=pd.read_csv(test_scores_path)
    train_dataset=build_dataset(train_features,labels_dict)
    valid_dataset = build_dataset(valid_features, labels_dict)
    test_dataset = build_dataset(test_features, labels_dict)

    return train_dataset,valid_dataset,test_dataset

if __name__=="__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        parser.add_argument("--metric", default="ndcg_cut_3")
        parser.add_argument("--col", default=DEFAULT_COL)
        parser.add_argument("--qpp_res_dir_base", default=DEFAULT_QPP_RES_DIR)
        parser.add_argument("--folds_file_name", default="folds_10.json")
        parser.add_argument("--features", nargs='+', default=DEFAULT_SELECTED_FEATURES)
        parser.add_argument('--penalty', default="l2")
        parser.add_argument("--res_dir", default=DEFAULT_RES_DIR)
        args = parser.parse_args()
        metric = args.metric
        col = args.col
        qpp_res_dir_base = args.qpp_res_dir_base
        folds_file_name = args.folds_file_name
        selected_features = args.features
        res_dir=args.res_dir
        penalty=args.penalty
        c_vals=DEFAULT_C
        EVAL_PATH = "/lv_local/home/tomergur/convo_search_project/data/eval/{}/{}".format(res_dir, col)
        RUNS_PATH = "/v/tomergur/convo/res/{}/{}".format(res_dir, col)
        runs, rewrites, rewrites_eval,turns_text = load_data(REWRITE_METHODS, EVAL_PATH, RUNS_PATH,col=col)
        label_dict = create_label_dict(rewrites_eval, metric)
        qpp_res_dir = "{}/{}/{}/".format(qpp_res_dir_base, res_dir, col)
        with open("{}/{}".format(qpp_res_dir, folds_file_name)) as f:
            folds = json.load(f)
        res=None
        for i,fold in enumerate(folds,start=1):
            train_dataset,valid_dataset,test_dataset=load_datasets(selected_features, label_dict, qpp_res_dir, i)
            normalizer = MinMaxScaler()
            x,labels=train_dataset
            x_norm = normalizer.fit_transform(x)
            pair_time = time.time()
            x_pair, y = create_pairwise_data(x_norm, labels)
            print("pairs creation time:", time.time() - pair_time)
            print(x_pair.shape)
            models = {c: (train_model((x_pair, y),c, args.penalty), normalizer) for c in
                      c_vals}
            model_perf = {}
            for c_val, model_tup in models.items():
                model, normalizer = model_tup
                valid_res = run_infrence(valid_dataset, model, normalizer)
                valid_perf = calc_metric(valid_res)
                model_perf[c_val] = valid_perf
            best_cval = max(model_perf.items(), key=lambda x: x[1])[0]
            print("best c val is:", best_cval)
            best_model, best_Normalizer = models[best_cval]
            fold_res = run_infrence(test_dataset, best_model, best_Normalizer)
            res= fold_res if res is None else res.append(fold_res)
        print(calc_metric(res))




