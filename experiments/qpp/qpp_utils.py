import pandas as pd
import json
import scipy
import itertools
import numpy as np
import math

def load_data(rewrite_methods, eval_path, runs_path, queries_field="first_stage_rewrites", col="cast19"):
    runs, rewrites, turns_text = {}, {}, {}

    rewrites_eval = load_eval(eval_path, rewrite_methods)

    # second_stage_queries
    for rewrite_method in rewrite_methods:
        input_path = "{}/{}_queries.json".format(runs_path, rewrite_method)
        with open(input_path) as f:
            rewrite_json = json.load(f)
        turns_text={k:v["query"] for k,v in rewrite_json.items() }
        rewrites[rewrite_method] = {qid: q_dict[queries_field][0] if queries_field in q_dict else q_dict["query"] for
                                    qid, q_dict in rewrite_json.items()}
    '''
    if "cast" in col:
        turns_text = get_cast_turns_text(col)
    else:
        turn_text = {}
    '''

    for rewrite_method in rewrite_methods:
        run_path = "{}/{}_run.txt".format(runs_path, rewrite_method)
        method_run = pd.read_csv(run_path, header=None, names=["qid", "Q0", "docid", "ranks", "score", "info"],
                                 delimiter="\t", dtype={"docid": str})
        runs[rewrite_method] = method_run

    return runs, rewrites, rewrites_eval, turns_text


def load_eval(eval_path, rewrite_methods):
    rewrites_eval = {}
    for rewrite_method in rewrite_methods:
        input_path = "{}/{}.txt".format(eval_path, rewrite_method)
        eval_list = pd.read_csv(input_path, header=None, names=["metric", "qid", "value"], delimiter="\t")
        eval_list = eval_list[~(eval_list.qid == "all")]
        eval_list = eval_list.assign(sid=eval_list.qid.map(lambda x: x.split("_")[0]))
        # eval_list.metric=eval_list.metric.str.strip()
        rewrites_eval[rewrite_method] = eval_list
    return rewrites_eval


def get_cast_turns_text(col):
    turns_text = {}
    turns_path = "./data/cast/evaluation_topics_j_2019.json" if col == "cast19" else "./data/cast/2020_automatic_evaluation_topics.json"
    with open(turns_path) as f:

        turns_json = json.load(f)
        for session in turns_json:
            session_num = str(session["number"])
            for turn_id, conversations in enumerate(session["turn"], start=1):
                qid = "{}_{}".format(session_num, turn_id)
                turns_text[qid] = conversations["raw_utterance"]
    return turns_text


def create_label_dict(rewrites_eval, metric):
    return {k: df[(df.metric.str.startswith(metric)) & (~(df.qid == "all"))].set_index("qid").value.to_dict() for k, df
            in rewrites_eval.items()}


def create_ctx(runs, rewrites, turns_text, col="cast19"):
    ctx = {}
    REWRITE_REF_LIST = ["t5", "all", "hqe", "quretec"]
    REWRITE_REF_LIST=["t5"]
    res_lists = {}
    for method_name, method_runs in runs.items():
        res_lists[method_name] = {qid: list(zip(q_run.docid.tolist(), q_run.score.tolist())) for qid, q_run in
                                  method_runs.groupby("qid")}
    for method_name, method_runs in runs.items():
        method_ctx = {}
        method_rewrites = rewrites[method_name]
        method_res_lists = res_lists[method_name]
        for qid, method_res_list in method_res_lists.items():
            q_ctx = {"res_list": method_res_list, "method": method_name, "turn_text": turns_text.get(qid)}
            sid, turn_id = qid.split("#") if col == "or_quac" else qid.split("_")
            prev_turns = []
            if col != "reddit":
                for i in range(int(turn_id)):
                    hist_qid = "{}#{}".format(sid, i) if col == "or_quac" else "{}_{}".format(sid, i + 1)
                    if hist_qid not in method_res_lists:
                        continue
                    prev_turns.append((method_rewrites[hist_qid],
                                       {"res_list": method_res_lists[hist_qid], "qid": hist_qid, "method": method_name,
                                        "turn_text": turns_text.get(hist_qid)}))
                q_ctx["history"] = prev_turns

            rewrites_ctx = []
            for ref_method in REWRITE_REF_LIST:
                if ref_method == method_name or qid not in res_lists[ref_method]:
                    continue
                rewrites_ctx.append((rewrites[ref_method][qid],
                                     {"res_list": res_lists[ref_method][qid], "qid": qid, "method": ref_method,
                                      "turn_text": turns_text.get(qid)}))
            q_ctx["ref_rewrites"] = rewrites_ctx
            q_ctx["qid"] = qid
            q_ctx["tid"] = int(turn_id)
            q_ctx["sid"] = sid
            method_ctx[qid] = q_ctx
        ctx[method_name] = method_ctx
    return ctx


def single_turn_corr(feature_values, labels,tid,corr_type="pearson"):
    qids=list(labels.keys())
    split_token = "#" if len(qids[0].split("#")) > 1 else "_"
    turn_labels={qid:v for qid,v in labels.items() if qid.split(split_token)[1]==tid}
    return calc_topic_corr(feature_values,turn_labels,corr_type)

def calc_topic_turn_corr(feature_values, labels, corr_type="pearson",min_queries=2):
    qids=list(labels.keys())
    split_token = "#" if len(qids[0].split("#")) > 1 else "_"
    turns_labels={}
    for qid in qids:
        cur_tid = qid.split(split_token)[1]
        if cur_tid not in turns_labels:
            turns_labels[cur_tid]={}
        turns_labels[cur_tid][qid]=labels[qid]
    #print({k:len(v) for k,v in turns_labels.items()})
    turn_corr=[calc_topic_corr(feature_values,turn_labels,corr_type=corr_type) for turn_labels in turns_labels.values() if len(turn_labels)>=min_queries]
    turn_corr=[x for x in turn_corr if not math.isnan(x)]
    #print(turn_corr)
    return np.mean(turn_corr)


def calc_topic_pairwise_acc(feature_values, labels, cmp_per_turn=True):
    num_pairs = 0
    true_pairs = 0
    qids = list(labels.keys())
    split_token = "#" if len(qids[0].split("#")) > 1 else "_"
    tids = {qid: int(qid.split(split_token)[1]) for qid in qids}
    for qid, qid2 in itertools.combinations(qids, 2):
        label = labels[qid]
        label2 = labels[qid2]
        if label == label2:
            continue
        #tid=int(qid[qid.rfind(split_token)+1:])
        #tid2 = int(qid2[qid2.rfind(split_token)+1:])
        #print(tid,tid2)
        #tid = int(qid.split(split_token)[1])
        #tid2 = int(qid2.split(split_token)[1])
        tid=tids[qid]
        tid2=tids[qid2]
        if cmp_per_turn and tid != tid2:
            continue
        num_pairs += 1
        feature = feature_values[qid]
        feature2 = feature_values[qid2]
        if (label > label2 and feature >= feature2) or (label < label2 and feature <= feature2):
            true_pairs += 1
    return round(true_pairs / num_pairs, 2), num_pairs


def calc_topic_corr(feature_values, labels, corr_type="pearson"):
    if corr_type not in ["pearson","kendall","spearman"]:
        raise("illegal corr type:",corr_type)
    qids = list(labels.keys())
    feature_lst = [feature_values[qid] for qid in qids]
    label_lst = [labels[qid] for qid in qids]
    if corr_type == "pearson":
        corr, p_val = scipy.stats.pearsonr(feature_lst, label_lst)
    elif corr_type == "kendall":
        corr, p_val = scipy.stats.kendalltau(feature_lst, label_lst)
    else:
        corr, p_val = scipy.stats.spearmanr(feature_lst, label_lst)
    return corr

def evaluate_topic_predictor(feature_values, labels, corr_type="pearson"):
    if corr_type=="TPA":
        return calc_topic_pairwise_acc(feature_values,labels)[0]
    elif corr_type=="PA":
        return calc_topic_pairwise_acc(feature_values, labels,False)[0]
    elif corr_type.startswith("turn_"):
        turn_corr_type=corr_type[5:]
        return calc_topic_turn_corr(feature_values, labels,turn_corr_type)
    elif corr_type.startswith("sturn_"):
        _,tid,s_turn_corr_type=corr_type.split("_")
        return single_turn_corr(feature_values,labels,tid,s_turn_corr_type)

    return calc_topic_corr(feature_values, labels,corr_type)



def topic_evaluate_extractor(extractor, rewrites, labels, ctx, return_raw_feature=False):
    #TODO: maybe use metaclass
    if hasattr(extractor,'calc_qpp_features'):
        feature_res=extractor.calc_qpp_features(rewrites,ctx)
    else:
        feature_res = {qid: extractor.calc_qpp_feature(q, **ctx[qid]) for qid, q in
                   rewrites.items() if qid in ctx}
    corr = calc_topic_corr(feature_res, labels)
    #corr,_ = calc_topic_pairwise_acc(feature_res, labels)
    if return_raw_feature:
        return corr, feature_res
    return corr
