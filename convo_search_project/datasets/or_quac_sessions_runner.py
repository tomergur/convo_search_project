import json
import time
from .utils import write_run
class ORQuacSessionRunner:
    def __init__(self,pipeline,add_canonical_rsp):
        self.pipeline=pipeline
        self.add_canonical_rsp=add_canonical_rsp
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
                canonical_response=None
                if self.add_canonical_rsp:
                    canonical_response=[t["answer"]["text"] if t["answer"]["text"]!="CANNOTANSWER" else None for t in conversation["history"]] if len(conversation["history"])>0 else None
                if args.log_queries:
                    run_res, query_dict = self.pipeline.retrieve(query, history=history,
                                                            qid=qid,tid=tid,canonical_rsp=canonical_response)
                    queries_dict[qid] = query_dict
                else:
                    run_res = self.pipeline.retrieve(query, history=history, qid=qid,tid=tid,canonical_rsp=canonical_response)
                write_run(f_out,qid,run_res)

                if args.log_lists:
                    runs[qid] = run_res
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
        return queries_dict, runs