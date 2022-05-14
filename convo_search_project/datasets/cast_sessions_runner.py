import json
import time
from .utils import output_run_file
import spacy
import numpy as np
import transformers
class CastSessionRunner:

    def __init__(self,pipeline,doc_fetcher):
        self.pipeline=pipeline
        self.doc_fetcher=doc_fetcher
        #, 'tagger', 'parser'
        self.english = spacy.load("en_core_web_sm", disable=['ner']) if doc_fetcher is not None else None
        model_name="bert-large-uncased-whole-word-masking-finetuned-squad"
        #self.qa_model=transformers.TFBertForQuestionAnswering.from_pretrained(model_name)
        #self.qa_tokenizer=transformers.AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline=transformers.pipeline("question-answering",model=model_name)

    def exctract_sentence_tf(self,canonical_response,next_turn_query,query):
        psg=self.english(canonical_response)
        sentences = [str(sent).strip() for sent in psg.sents]
        #if t.pos_ in ['VERB','NOUN','ADJ']
        next_turn_set=set([t.lemma_ for t in self.english(next_turn_query)  if not (t.is_stop or t.is_punct) ])
        sentences_tokens=[set([t.lemma_ for t in self.english(s)  if  not (t.is_stop or t.is_punct) ]) for s in sentences]
        next_turn_scores=[len(next_turn_set.intersection(st)) for st in sentences_tokens]
        if max(next_turn_scores) >0:
            best_score_pos=np.argmax(next_turn_scores)
            return sentences[best_score_pos]
        cur_turn_set = set([t.lemma_ for t in self.english(query) if not (t.is_stop or t.is_punct)])
        cur_turn_scores = [len(cur_turn_set.intersection(st)) for st in sentences_tokens]
        if max(cur_turn_scores) >0:
            best_score_pos=np.argmax(cur_turn_scores)
            return sentences[best_score_pos]
        return ""

    def exctract_sentence(self,canonical_response,next_turn_query,query):
        '''
        ret = self.qa_tokenizer(query,
                             canonical_response, max_length=512,
                             truncation=True, padding="max_length", return_token_type_ids=True, return_tensors='tf')
        input_ids = ret['input_ids']
        tt_ids = ret['token_type_ids']
        att_mask = ret['attention_mask']
        output = self.qa_model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                             training=False)
        '''
        output=self.qa_pipeline(question=query,context=canonical_response)
        return output["answer"]
    def run_sessions(self,args):
        input_queries_file = args.input_queries_file
        query_field = "manual_rewritten_utterance" if args.use_manual_run else "raw_utterance"
        with open(input_queries_file) as json_file:
            data = json.load(json_file)
        runs = {}
        queries_dict = {}
        for session in data:
            start_time = time.time()
            session_num = str(session["number"])
            history = []
            canonical_response = [] if self.doc_fetcher else None
            for turn_id, conversations in enumerate(session["turn"],start=1):
                query_start_time = time.time()
                query = conversations[query_field]
                conversation_num = str(conversations["number"])
                qid = session_num + "_" + conversation_num
                print(qid, query)
                if args.log_queries:
                    run_res, query_dict = self.pipeline.retrieve(query, history=history, canonical_rsp=canonical_response,
                                                            qid=qid,tid=conversation_num)
                    queries_dict[qid] = query_dict
                else:
                    run_res = self.pipeline.retrieve(query, history=history, canonical_rsp=canonical_response, qid=qid,tid=conversation_num)
                history.append(query)
                if self.doc_fetcher and turn_id<len(session["turn"]):
                    rsp_doc = self.doc_fetcher.doc(conversations["manual_canonical_result_id"])
                    raw_canoical_rsp=rsp_doc.raw()
                    next_turn_query=session["turn"][turn_id][query_field]
                    canonical_rsp=self.exctract_sentence(raw_canoical_rsp,next_turn_query,query)
                    canonical_response.append(canonical_rsp)
                runs[qid] = run_res
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
            print("session {} runtime is:{} sec".format(session_num, time.time() - start_time))
        run_output_file = "{}/{}_run.txt".format(args.output_dir, args.run_name)
        output_run_file(run_output_file, runs)
        return queries_dict, runs

