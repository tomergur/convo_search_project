import json
import time
from .utils import write_run
class ORQuacSessionRunner:
    def __init__(self,pipeline,doc_fetcher):
        self.pipeline=pipeline
        self.doc_fetcher=doc_fetcher
    def run_sessions(self,args):
        input_queries_file = args.input_queries_file
        query_field = "rewrite" if args.use_manual_run else "question"
        runs = {}
        queries_dict = {}
        run_output_file = "{}/{}_run.txt".format(args.output_dir, args.run_name)
        with open(input_queries_file) as json_file,open(run_output_file,'w') as f_out:
            for i,session_query_line in enumerate(json_file.readlines()):
                conversation=json.loads(session_query_line)
                query_start_time = time.time()
                query = conversation[query_field]
                conversation_num,tid= conversation["qid"].split("#")
                qid = conversation["qid"]
                print(i,qid,conversation_num,tid, query)
                history=[t["question"] for t in conversation["history"]]
                if args.log_queries:
                    run_res, query_dict = self.pipeline.retrieve(query, history=history,
                                                            qid=qid,tid=tid)
                    queries_dict[qid] = query_dict
                else:
                    run_res = self.pipeline.retrieve(query, history=history, qid=qid,tid=tid)
                write_run(f_out,qid,run_res)
                #TODO: add or remove canonical response
                '''
                if self.doc_fetcher:
                    rsp_doc = self.doc_fetcher.doc(conversations["manual_canonical_result_id"])
                    canonical_response.append(rsp_doc.raw())
                '''
                if args.log_lists:
                    runs[qid] = run_res
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
        return queries_dict, runs