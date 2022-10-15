from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import json
import tensorflow as tf
from keras import backend as K
from .supervised.groupwise_model import GroupwiseBert

def truncate_query(query,tokenizer, max_length=128):
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
            print("truncated q:", truncated_query)
            return truncated_query
    assert (False)
    return query

def get_passage(searcher,psg_id):
    doc = searcher.doc(psg_id)
    raw_doc = doc.raw()
    try:
        passage = json.loads(raw_doc)["contents"]
    except ValueError as e:
        passage = raw_doc
    return passage

def modify_query(query,ctx,append_history,append_prev_turns):
    if (append_history or append_prev_turns) and len(ctx["history"]) > 0:
        if append_history:
            prev_raw_turn = [turn[1]["turn_text"] for turn in ctx["history"]]
        else:
            prev_raw_turn = [turn[0] for turn in ctx["history"]]
        return " [SEP] ".join(prev_raw_turn + [query])
    return query

class BertQPP:
    def __init__(self,searcher,model_name_or_path_pattern,col,append_history=False,append_prev_turns=False):
        self.searcher = searcher
        self.model_name_or_path_pattern = model_name_or_path_pattern
        self.col = col
        self.method = None
        self.append_history=append_history
        self.i=0
        self.append_prev_turns=append_prev_turns
        assert(not(append_history and append_prev_turns))


    def calc_qpp_feature(self, query, **ctx):
        if self.i%100==0:
            print("bert qpp:",self.i)
        self.i+=1
        cur_method=ctx["method"]
        if cur_method!=self.method:
            model_path=self.model_name_or_path_pattern.format(self.col,cur_method)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
            self.method=cur_method
        res_list = ctx["res_list"]
        top_doc_id=res_list[0][0]
        passage = get_passage(self.searcher,top_doc_id)
        query=modify_query(query,ctx,self.append_history,self.append_prev_turns)
        return self.calc_score(query,passage)

    def calc_score(self,query,passage):
        trunc_query=truncate_query(query,self.tokenizer)
        ret = self.tokenizer(trunc_query,passage, max_length=512,truncation=True, return_token_type_ids=True, return_tensors='tf')

        input_ids = ret['input_ids']
        tt_ids = ret['token_type_ids']
        att_mask = ret['attention_mask']
        logits = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                            training=False)[0]
        #print("logits",logits)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits)
        scores=scores.numpy()
        #print(scores)
        res = scores[0,-1]
        #print("query res",res,type(res.item()))
        return float(res.item())





class GroupwiseBertQPP:
    def __init__(self, searcher, text_model_path_pattern,group_model_path_pattern, col, append_history=False, append_prev_turns=False):
        self.searcher = searcher
        self.text_model_path_pattern = text_model_path_pattern
        self.group_model_path_pattern=group_model_path_pattern
        self.col = col
        self.method = None
        self.append_history = append_history
        self.append_prev_turns = append_prev_turns
        self.i = 0

        assert (not (append_history and append_prev_turns))


    def calc_group_scores(self,queries,passages):
        trunc_queries = [truncate_query(query,self.tokenizer) for query in queries]
        ret = self.tokenizer(trunc_queries, passages, max_length=512, truncation=True, return_token_type_ids=True,
                             return_tensors='tf',padding=True)
        logits = self.model(ret,training=False)
        scores = tf.keras.layers.Activation(tf.nn.softmax)(logits) if logits.shape[1]>1 else logits
        scores = scores.numpy()
        return [s.item() for s in scores[:,-1]]

    def calc_qpp_features(self,queries, ctx):
        qids=list(queries.keys())
        sids=set([c['sid'] for c in ctx.values()])
        sessions={sid:[] for sid in sids}
        for qid in qids:
            sid=ctx[qid]['sid']
            sessions[sid].append(qid)

        cur_method=ctx[qids[0]]["method"]
        if cur_method!=self.method:
            text_model_path = self.text_model_path_pattern.format(self.col,cur_method)
            group_model_path = self.group_model_path_pattern.format(self.col,cur_method)
            K.clear_session()
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_path, use_fast=True)
            self.model = GroupwiseBert.from_pretrained(text_model_path,group_model_path)
            self.model.group_bert.summary()
            self.method=cur_method

        res={}
        for sid,s_qids in sessions.items():
            s_qids.sort()
            #print(sid,s_qids)
            if self.i % 100 == 0:
                print("group bert qpp:", self.i)
            self.i += 1
            s_queries=[modify_query(queries[qid], ctx[qid], self.append_history, self.append_prev_turns) for qid in s_qids]
            passages=[get_passage(self.searcher,ctx[qid]['res_list'][0][0]) for qid in s_qids]
            scores=self.calc_group_scores(s_queries,passages)
            #print(scores,len(scores))
            for qid,score in zip(s_qids,scores):
                res[qid]=score
        return res












