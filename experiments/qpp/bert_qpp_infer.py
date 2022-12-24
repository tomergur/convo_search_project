from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import json
import tensorflow as tf
from keras import backend as K
from .supervised.groupwise_model import GroupwiseBert
from .supervised.bert_pl import BertPL
from .supervised.groupwise_bert_pl import GroupwiseBertPL
from .supervised.seq_qpp_model import SeqQPP


def truncate_query(query, tokenizer, max_length=128):
    q_tokens = tokenizer(query)
    q_len = len(q_tokens['input_ids'])
    if max_length >= q_len:
        return query
    query_words = query.split(" ")
    for i in range(1, len(query_words)):
        truncated_query = " ".join(query_words[i:])
        q_tokens = tokenizer(truncated_query)
        q_len = len(q_tokens['input_ids'])
        if max_length >= q_len:
            # print("truncated q:", truncated_query)
            return truncated_query
    assert (False)
    return query


def get_passage(searcher, psg_id):
    doc = searcher.doc(psg_id)
    raw_doc = doc.raw()
    try:
        passage = json.loads(raw_doc)["contents"]
    except ValueError as e:
        passage = raw_doc
    return passage


def modify_query(query, ctx, append_history, append_prev_turns):
    if (append_history or append_prev_turns) and len(ctx["history"]) > 0:
        if append_history:
            prev_raw_turn = [turn[1]["turn_text"] for turn in ctx["history"]]
        else:
            prev_raw_turn = [turn[0] for turn in ctx["history"]]
        return " [SEP] ".join(prev_raw_turn + [query])
    return query


import random


class SingleTurnBertQPP:
    def __init__(self, searcher, model_name_or_path_pattern, col, append_history=False, append_prev_turns=False):
        self.searcher = searcher
        self.model_name_or_path_pattern = model_name_or_path_pattern
        self.col = col
        self.append_history = append_history
        self.i = 0
        self.append_prev_turns = append_prev_turns
        assert (not (append_history and append_prev_turns))

    def calc_qpp_features(self, queries, ctx):
        qids = list(queries.keys())
        turn_qids = {}
        for qid in qids:
            tid = ctx[qid]['tid']
            cur_turn = turn_qids.get(tid, [])
            cur_turn.append(qid)
            turn_qids[tid] = cur_turn
        cur_method = ctx[qids[0]]["method"]
        res = {qid: random.random() for qid in qids}
        col_add = 1 if self.col == "or_quac" else 0
        for tid, t_qids in turn_qids.items():
            if tid + col_add not in [1, 2, 3, 5, 7, 9]:
                continue
            print("run turn", tid)
            model_path = self.model_name_or_path_pattern.format(self.col, cur_method, tid + col_add)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            for qid in t_qids:
                res[qid] = self.calc_qpp_feature(queries[qid], **ctx[qid])
        return res

    def calc_score(self, query, passage):
        trunc_query = truncate_query(query, self.tokenizer)
        ret = self.tokenizer(trunc_query, passage, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf')

        input_ids = ret['input_ids']
        tt_ids = ret['token_type_ids']
        att_mask = ret['attention_mask']
        logits = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                            training=False)[0]
        # print("logits",logits)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1] > 1 else logits
        scores = scores.numpy()
        # print(scores)
        res = scores[0, -1]
        # print("query res",res,type(res.item()))
        return float(res.item())

    def calc_qpp_feature(self, query, **ctx):
        if self.i % 100 == 0:
            print("bert qpp:", self.i)
        self.i += 1
        res_list = ctx["res_list"]
        top_doc_id = res_list[0][0]
        passage = get_passage(self.searcher, top_doc_id)
        query = modify_query(query, ctx, self.append_history, self.append_prev_turns)
        return self.calc_score(query, passage)


class RewritesBertQPP:
    REWRITE_METHODS = ['t5', 'all', 'hqe', 'quretec']

    def __init__(self, searcher, model_path,top_docs=1, append_history=False, append_prev_turns=False):
        self.searcher = searcher
        self.model_path=model_path
        self.model=None
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.append_history = append_history
        self.append_prev_turns = append_prev_turns
        self.i = 0
        self.top_docs=top_docs
        self.cache = {k: {} for k in RewritesBertQPP.REWRITE_METHODS}

    def calc_group_scores(self, queries, passages):
        trunc_queries = [truncate_query(query, self.tokenizer) for query in queries]
        ret = self.tokenizer(trunc_queries, passages, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf', padding=True)
        ret = {k: tf.expand_dims(v, 0) for k, v in ret.items()}
        logits = self.model(ret, training=False)
        if len(logits.shape) == 0:
            return [logits.numpy().item()]
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1] > 1 else logits
        scores = tf.squeeze(scores, 0) if len(tf.shape(scores)) > 2 else scores
        scores = scores.numpy()
        return [s.item() for s in scores[:, -1]]

    def calc_qpp_feature(self, query, **ctx):
        #TODO: fix lazy evaluation?????
        if self.model is None:
            K.clear_session()
            if self.top_docs==1:
                self.model = GroupwiseBert.from_pretrained(self.model_path, output_mode="tokens")
            else:
                self.model=GroupwiseBertPL.from_pretrained(self.model_path,self.top_docs)
        if self.i % 100 == 0:
            print("bert qpp:", self.i)
        self.i += 1
        cur_method = ctx["method"]
        qid = ctx["qid"]
        if qid in self.cache[cur_method]:
            return self.cache[cur_method][qid]
        res_list = ctx["res_list"]
        top_doc_ids = [x[0] for x in res_list[:self.top_docs]]
        rewrites = {cur_method: (query, top_doc_ids)}
        for rewrite, rewrite_ctx in ctx['ref_rewrites']:
            res_list = rewrite_ctx["res_list"]
            top_doc_ids = [x[0] for x in res_list[:self.top_docs]]
            rewrites[rewrite_ctx['method']] = (rewrite, top_doc_ids)
        queries,passages=[],[]
        for rewrite_method in RewritesBertQPP.REWRITE_METHODS:
            queries += [rewrites[rewrite_method][0]]*self.top_docs
            passages += [get_passage(self.searcher, psg) for psg in rewrites[rewrite_method][1]]
        res = self.calc_group_scores(queries, passages)
        for i, rewrite_method in enumerate(RewritesBertQPP.REWRITE_METHODS):
            self.cache[rewrite_method][qid] = res[i]
        return self.cache[cur_method][qid]


class BertQPP:
    def __init__(self, searcher, model_name_or_path_pattern, col, append_history=False, append_prev_turns=False):
        self.searcher = searcher
        self.model_name_or_path_pattern = model_name_or_path_pattern
        self.col = col
        self.method = None
        self.append_history = append_history
        self.i = 0
        self.append_prev_turns = append_prev_turns
        assert (not (append_history and append_prev_turns))
        self.batch_size = 8

    def calc_qpp_feature(self, query, **ctx):
        if self.i % 100 == 0:
            print("bert qpp:", self.i)
        self.i += 1
        cur_method = ctx["method"]
        if cur_method != self.method:
            model_path = self.model_name_or_path_pattern.format(self.col, cur_method)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.method = cur_method
        res_list = ctx["res_list"]
        top_doc_id = res_list[0][0]
        passage = get_passage(self.searcher, top_doc_id)
        query = modify_query(query, ctx, self.append_history, self.append_prev_turns)
        return self.calc_score(query, passage)

    def calc_qpp_features(self, queries, ctx):
        qids = list(queries.keys())
        cur_method = ctx[qids[0]]["method"]
        if cur_method != self.method:
            model_path = self.model_name_or_path_pattern.format(self.col, cur_method)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.method = cur_method
        queries_text = []
        passages = []
        for qid in qids:
            query = queries[qid]
            q_ctx = ctx[qid]
            res_list = q_ctx["res_list"]
            top_doc_id = res_list[0][0]
            passage = get_passage(self.searcher, top_doc_id)
            query = modify_query(query, q_ctx, self.append_history, self.append_prev_turns)
            queries_text.append(query)
            passages.append(passage)
        ret = self.tokenizer(queries_text, passages, max_length=512, truncation=True, padding="max_length",
                             return_token_type_ids=True, return_tensors='tf')
        pairs_ds = tf.data.Dataset.from_tensor_slices(
            {"input_ids": ret['input_ids'], "attention_mask": ret['attention_mask'],
             "token_type_ids": ret['token_type_ids']})
        logits = self.model.predict(pairs_ds.batch(self.batch_size), batch_size=self.batch_size, verbose=1).logits
        tf.keras.layers.Activation(tf.nn.softmax)(logits)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1] > 1 else logits
        scores = scores.numpy()
        res = {qids[i]: scores[i, -1].item() for i in range(len(qids))}
        return res

    def calc_score(self, query, passage):
        trunc_query = truncate_query(query, self.tokenizer)
        ret = self.tokenizer(trunc_query, passage, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf')

        input_ids = ret['input_ids']
        tt_ids = ret['token_type_ids']
        att_mask = ret['attention_mask']
        logits = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                            training=False)[0]
        # print("logits",logits)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1] > 1 else logits
        scores = scores.numpy()
        # print(scores)
        res = scores[0, -1]
        # print("query res",res,type(res.item()))
        return float(res.item())


class BertPLQPP:

    def __init__(self, searcher, model_path_pattern, col, chunk_size,top_docs=10):
        self.searcher = searcher
        self.model_path_pattern = model_path_pattern
        self.col = col
        self.i = 0
        self.chunk_size = chunk_size
        self.method = None
        self.top_docs=top_docs

    def calc_qpp_feature(self, query, **ctx):
        if self.i % 100 == 0:
            print("bert qpp:", self.i)
        self.i += 1
        cur_method = ctx["method"]
        if cur_method != self.method:
            model_path = self.model_path_pattern.format(self.col, cur_method)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = BertPL.from_pretrained(model_path, self.chunk_size)
            self.method = cur_method
        passages = [get_passage(self.searcher, ctx['res_list'][i][0]) for i in range(self.top_docs)]
        query = [modify_query(query, ctx, False, False)] * len(passages)
        return self.calc_query_score(query, passages)

    def calc_query_score(self, queries, passages):
        trunc_queries = [truncate_query(query, self.tokenizer) for query in queries]
        ret = self.tokenizer(trunc_queries, passages, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf', padding=True)
        ret = {k: tf.expand_dims(v, 0) for k, v in ret.items()}
        logits = self.model(ret, training=False)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits)
        #print("scores", scores.numpy())
        num_rel = tf.math.argmax(scores, axis=-1).numpy()
        #print("num rel", num_rel)
        return sum([s.item() / i for i, s in enumerate(num_rel, start=1)])
        return [s.item() for s in scores[:, -1]]


class GroupwiseBertQPP:
    # TODO modify infer parameter
    TOP_DOCS = 10

    def __init__(self, searcher, model_path_pattern, col, append_history=False,
                 append_prev_turns=False, infer_mode="dialogue", group_agg_func=None, output_mode=None,
                 max_seq_length=None, seqQPP=False):
        self.searcher = searcher
        self.model_path_pattern = model_path_pattern
        self.col = col
        self.method = None
        self.append_history = append_history
        self.append_prev_turns = append_prev_turns
        self.i = 0
        assert (infer_mode in [None, "dialogue", "query"])
        self.infer_mode = infer_mode
        self.group_agg_func = group_agg_func
        self.output_mode = output_mode
        self.max_seq_length = max_seq_length
        assert (not (append_history and append_prev_turns))
        self.seqQPP = seqQPP

    def calc_group_scores(self, queries, passages):
        trunc_queries = [truncate_query(query, self.tokenizer) for query in queries]
        ret = self.tokenizer(trunc_queries, passages, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf', padding=True)
        ret = {k: tf.expand_dims(v, 0) for k, v in ret.items()}
        logits = self.model(ret, training=False)
        if len(logits.shape) == 0:
            return [logits.numpy().item()]
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1] > 1 else logits
        scores = tf.squeeze(scores, 0) if len(tf.shape(scores)) > 2 else scores
        scores = scores.numpy()
        return [s.item() for s in scores[:, -1]]

    def query_infer_mode(self, queries, ctx):
        res = {}
        for qid, query in queries.items():
            q_ctx = ctx[qid]
            m_query = modify_query(query, q_ctx, self.append_history, self.append_prev_turns)
            # print(qid,m_query)
            passages = [get_passage(self.searcher, q_ctx['res_list'][i][0]) for i in range(GroupwiseBertQPP.TOP_DOCS)]
            # print(qid,passages)
            score = self.calc_group_scores([m_query] * len(passages), passages)
            # print("query groupwise score",score)
            if self.i % 100 == 0:
                print("************group bert qpp:", self.i)
            self.i += 1
            res[qid] = score[0]
        return res

    def calc_qpp_features(self, queries, ctx):
        cur_method = list(ctx.values())[0]["method"]
        if cur_method != self.method:
            model_path = self.model_path_pattern.format(self.col, cur_method)
            K.clear_session()
            # self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            # TODO: change tokenizer path
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
            if self.seqQPP:
                self.model = SeqQPP.from_pretrained(model_path)
            else:
                self.model = GroupwiseBert.from_pretrained(model_path,
                                                           group_agg_func=self.group_agg_func,
                                                           output_mode=self.output_mode,
                                                           max_seq_length=self.max_seq_length)
            self.method = cur_method
        if self.infer_mode == "dialogue":
            return self.dialogue_infer_mode(queries, ctx)
        return self.query_infer_mode(queries, ctx)

    def dialogue_infer_mode(self, queries, ctx):
        qids = list(queries.keys())
        sids = set([c['sid'] for c in ctx.values()])
        sessions = {sid: [] for sid in sids}
        for qid in qids:
            sid = ctx[qid]['sid']
            sessions[sid].append(qid)
        res = {}
        for sid, s_qids in sessions.items():
            s_qids.sort()
            # print(sid,s_qids)
            if self.i % 100 == 0:
                print("group bert qpp:", self.i)
            self.i += 1
            s_queries = [modify_query(queries[qid], ctx[qid], self.append_history, self.append_prev_turns) for qid in
                         s_qids]
            passages = [get_passage(self.searcher, ctx[qid]['res_list'][0][0]) for qid in s_qids]
            scores = self.calc_group_scores(s_queries, passages)
            # print(scores,len(scores))
            for qid, score in zip(s_qids, scores):
                res[qid] = score
        return res
