import json
import time
from .utils import write_run
class TopioCQASessionsRunner:
    def __init__(self, pipeline, add_canonical_rsp):
        self.pipeline = pipeline
        self.add_canonical_rsp = add_canonical_rsp

    def run_sessions(self, args):
        input_queries_file = args.input_queries_file
        query_field = "Question"
        runs = {}
        queries_dict = {}
        run_output_file = "{}/{}_run.txt".format(args.output_dir, args.run_name)
        with open(input_queries_file) as json_file, open(run_output_file, 'w') as f_out:
            conversations = json.load(json_file)
            history = []
            canonical_response = [] if self.add_canonical_rsp else None
            for i, conversation in enumerate(conversations):
                query_start_time = time.time()
                query = conversation[query_field]
                conversation_num, tid = conversation["Conversation_no"], conversation["Turn_no"]
                qid = "{}_{}".format(conversation_num, tid)
                print(i, qid, conversation_num, tid, query)
                if tid == 1:
                    history = []
                    canonical_response = [] if self.add_canonical_rsp else None
                if args.log_queries:
                    run_res, query_dict = self.pipeline.retrieve(query, history=history,
                                                                 qid=qid, tid=tid, canonical_rsp=canonical_response)
                    queries_dict[qid] = query_dict
                else:
                    run_res = self.pipeline.retrieve(query, history=history, qid=qid, tid=tid,
                                                     canonical_rsp=canonical_response)
                write_run(f_out, qid, run_res)
                if args.log_lists:
                    runs[qid] = run_res
                history.append(query)
                if self.add_canonical_rsp:
                    canonical_response.append(conversation["Answer"])
                print("query {} runtime is:{} sec".format(qid, time.time() - query_start_time))
        return queries_dict, runs