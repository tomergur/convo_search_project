from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import json
import tensorflow as tf
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
        top_doc = self.searcher.doc(top_doc_id)
        raw_doc = top_doc.raw()
        try:
            passage= json.loads(raw_doc)["contents"]
        except ValueError as e:
            passage=raw_doc
        if (self.append_history or self.append_prev_turns) and len(ctx["history"])>0:
            if self.append_history:
                prev_raw_turn=[turn[1]["turn_text"] for turn in ctx["history"]]
            else:
                prev_raw_turn = [turn[0] for turn in ctx["history"]]
            query=" [SEP] ".join(prev_raw_turn+[query])
            #print("appended hist:",query)
        return self.calc_score(query,passage)

    def calc_score(self,query,passage):
        trunc_query=self.truncate_query(query)
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



    def truncate_query(self,query,max_length=128):
        q_tokens = self.tokenizer(query)
        q_len = len(q_tokens['input_ids'])
        if max_length >= q_len:
            return query
        '''
        query_turns = query.split("[SEP]")
        for i in range(1, len(query_turns)):
            truncated_query = "[SEP]".join(query_turns[i:])
            q_tokens = self.tokenizer(truncated_query)
            q_len = len(q_tokens['input_ids'])
            if max_length >= q_len:
                print("truncated q:", truncated_query)
                return truncated_query
        '''
        query_words = query.split(" ")
        for i in range(1, len(query_words)):
            truncated_query = " ".join(query_words[i:])
            q_tokens = self.tokenizer(truncated_query)
            q_len = len(q_tokens['input_ids'])
            if max_length >= q_len:
                print("truncated q:", truncated_query)
                return truncated_query
        assert (False)
        return query
