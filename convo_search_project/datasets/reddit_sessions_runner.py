import json
import time
from .utils import write_run

class RedditSessionRunner:
    def __init__(self,pipeline):
        self.pipeline=pipeline
    def run_sessions(self,args):
        input_queries_file = args.input_queries_file
        runs = {}
        queries_dict = {}
        run_output_file = "{}/{}_run.txt".format(args.output_dir, args.run_name)
        with open(input_queries_file,encoding='utf-8') as json_file, open(run_output_file, 'w') as f_out:
            for i,session_json in enumerate(json_file.readlines()):
                session=json.loads(session_json)
                qid=session["id"]
                query_start_time = time.time()
                query = session["target"]["body"] if "target" in session else session["gold"]["body"]
                history=[session["title"]]+[turn["body"] for turn in session["context"] if len(turn["body"])>0]
                print(i, qid, query)
                if args.log_queries:
                    run_res, query_dict = self.pipeline.retrieve(query, history=history,
                                                            qid=qid,canonical_rsp=None)
                    queries_dict[qid] = query_dict
                else:
                    run_res = self.pipeline.retrieve(query, history=history, qid=qid,canonical_rsp=None)
                write_run(f_out,qid,run_res)
                if args.log_lists:
                    runs[qid] = run_res
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
            return queries_dict, runs

