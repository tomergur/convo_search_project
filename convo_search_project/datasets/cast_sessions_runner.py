import json
import time

class CastSessionRunner:
    def __init__(self,pipeline,doc_fetcher):
        self.pipeline=pipeline
        self.doc_fetcher=doc_fetcher
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
            for turn_id, conversations in enumerate(session["turn"]):
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
                if self.doc_fetcher:
                    rsp_doc = self.doc_fetcher.doc(conversations["manual_canonical_result_id"])
                    canonical_response.append(rsp_doc.raw())
                runs[qid] = run_res
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
            print("session {} runtime is:{} sec".format(session_num, time.time() - start_time))
        return queries_dict, runs

