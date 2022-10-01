import argparse
import json
import time

from pyserini.search import SimpleSearcher
import tensorflow as tf
import os
from convo_search_project.pipeline import Pipeline
from convo_search_project.rerankers import BertReranker,Bm25Reranker
from convo_search_project.rerankers import JaacardReranker
from convo_search_project.datasets import CastSessionRunner,ORQuacSessionRunner,QreccSessionRunner,TopioCQASessionsRunner,RedditSessionRunner
from convo_search_project.runs_cache import RunsCache
from convo_search_project.doc_modify import modify_to_all_queries,modify_to_single_queries,modify_to_append_all_queries
# TODO: delete
from convo_search_project.mono_bert import MonoBERT

from convo_search_project.rewriters import create_rewriter
from pyserini.index import IndexReader

def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='running retrieval pipeline for our convo search project')
    parser.add_argument('--run_name', required=True,
                        help='the name of the current run. will be the name of the output files')
    parser.add_argument('--collection_type',default="cast")
    parser.add_argument('--input_queries_file', required=True, help='input queries file')
    parser.add_argument('--output_dir', required=True, help='output dir for the run')
    parser.add_argument('--count', type=int, default=1000,
                        help="number of result retrieved in the first stage retrieval")
    parser.add_argument("--log_queries", action='store_true', default=True,
                        help="log the queries to a file in the output dir ")
    parser.add_argument("--log_lists", action='store_true', default=False,
                        help="log the retrieved list including the text")

    parser.add_argument("--first_stage_ranker", default="bm25", help="select the first stage ranking function")
    parser.add_argument('--tpu', help='name of tpu device')
    parser.add_argument("--use_manual_run", action='store_true', default=False,
                        help='use the manually rewritten queries')

    # Parameters for BM25.
    parser.add_argument('--k1', type=float, default=0.82, help='BM25 k1 parameter')
    parser.add_argument('--b', type=float, default=0.68, help='BM25 b parameter')

    #parmeters for kld
    parser.add_argument("--mu",type=int,default=1000)

    # parameters for cached first stage retreival
    parser.add_argument("--first_stage_cache_path", help="the path to the first stage cache file/s")

    # index path for dense retrieval
    parser.add_argument("--index_path")

    # rewriter params
    parser.add_argument("--rewriters", nargs="+", default=[], help='list of rewriters')
    parser.add_argument("--second_stage_rewriters", nargs="+",
                        help="list of rewriters for the second stage retrieval")

    # canonical response support
    parser.add_argument('--add_canonical_response', default=False, action='store_true', help='add canonical response')

    # T5 rewriter
    parser.add_argument('--T5_model_str', help='t5 rewriter model str')
    parser.add_argument('--T5_num_queries_generated', type=int, help='number of queries generated by the t5 model')
    parser.add_argument('--T5_rewriter_context_window', type=int, help="limit the context window of the t5 rewriter")
    parser.add_argument('--T5_rewriter_selected_query_rank', type=int,
                        help="start using the t5 output from an offset and not the best query")
    parser.add_argument('--T5_rewriter_sliding_window_fusion', action='store_true',
                        help='use sliding window fusion mode')
    parser.add_argument('--T5_append_history',action='store_true')
    # file rewriter
    parser.add_argument('--queries_rewrites_path', help='path to query rewrites cached in csv')

    #use [SEP] token for each turn
    parser.add_argument('--use_sep_token',action='store_true',default=False)
    # prev turns rewriter
    parser.add_argument('--prev_turns', type=int)

    # QuReTeC parameters
    parser.add_argument('--quretec_model_path', help='model path for QuReTec')

    # parameters for reranker
    parser.add_argument('--rerank', action='store_true', default=False, help='rerank BM25 output using BERT')
    parser.add_argument('--reranker_batch_size', type=int, default=32, help='reranker batch size for inference')
    parser.add_argument('--reranker_type', default='bert', help='select the reranker type')
    parser.add_argument('--reranker_model_path',default="castorini/monobert-large-msmarco-finetune-only")
    parser.add_argument('--reranker_device', help='reranker device to use')
    parser.add_argument('--is_tf',action='store_true')

    #doc modfier
    parser.add_argument("--modify_documents_func")
    parser.add_argument("--doc2q_path")

    #jaccard
    parser.add_argument('--jaccard_func')

    # Return args
    args = parser.parse_args()
    return args

# output the intial list files as json
def output_as_initial_list_as_single_file(output_path, runs, doc_searcher):
    res = {}
    for qid, run_res in runs.items():
        q_res = []
        for rank, doc in enumerate(run_res, start=1):
            doc_dict = {}
            doc_dict['docid'] = doc.docid
            doc_dict['rank'] = rank
            doc_dict['score'] = float(doc.score)
            if doc_searcher:
                doc_content = doc_searcher.doc(doc.docid)
                if doc_content:
                    doc_dict['content'] = doc_content.raw()
            else:
                doc_dict['content'] = doc.raw
            q_res.append(doc_dict)
        res[qid] = q_res
    with open(output_path, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=True)

def output_as_initial_list(run_name,output_dir, runs, doc_searcher):
    if len(runs)<=300:
        lists_output_file = "{}/{}_lists.json".format(args.output_dir, args.run_name)
        output_as_initial_list_as_single_file(lists_output_file, runs, doc_searcher)
        return
    lists_dir="{}/{}".format(output_dir,run_name)
    if not os.path.exists(lists_dir):
        os.mkdir(lists_dir)
    for qid, run_res in runs.items():
        q_res = []
        for rank, doc in enumerate(run_res, start=1):
            doc_dict = {}
            doc_dict['docid'] = doc.docid
            doc_dict['rank'] = rank
            doc_dict['score'] = float(doc.score)
            if doc_searcher:
                doc_content = doc_searcher.doc(doc.docid)
                if doc_content:
                    doc_dict['content'] = doc_content.raw()
            else:
                doc_dict['content'] = doc.raw
            q_res.append(doc_dict)
        with open("{}/{}.json".format(lists_dir,qid), "w") as f:
            json.dump(q_res, f, ensure_ascii=False, indent=True)


def output_queries_file(output_path, quries_dict):
    with open(output_path, "w") as f:
        json.dump(quries_dict, f, indent=True)

def build_reranker(
        args, device=None, strategy=None):
    """Returns a  reranker using the provided model name or path to load from"""
    if args.reranker_type == "bm25":
        #TODO: remove constant path and maybe adapt to or_quac
        INDEX_PATH="/lv_local/home/tomergur/.cache/pyserini/indexes/index-cast2019.36e604d7f5a4e08ade54e446be2f6345/"
        searcher=collection_type_to_searcher(args.collection_type)
        searcher.set_bm25(args.k1,args.b)
        if args.collection_type=="cast":
            index_reader=IndexReader.from_prebuilt_index("cast2019")
        else:
            index_reader=None
        return Bm25Reranker(index_reader,searcher.get_similarity())
    elif args.reranker_type=="jaccard":
        return JaacardReranker(args.jaccard_func if 'jaccard_func' in args else None)
    name_or_path=args.reranker_model_path
    from_pt= True if 'is_tf' not in args else False
    model = BertReranker.get_model(name_or_path, from_pt=from_pt)
    tokenizer = BertReranker.get_tokenizer(name_or_path)
    return BertReranker(model, tokenizer, batch_size=args.reranker_batch_size, device=device, strategy=strategy)

def build_bert_reranker2(
        name_or_path: str = "castorini/monobert-large-msmarco-finetune-only",
        batch_size: int = 32, device: str = None):
    """Returns a BERT reranker using the provided model name or path to load from"""
    model = MonoBERT.get_model(name_or_path, device=device)
    tokenizer = MonoBERT.get_tokenizer(name_or_path)
    return MonoBERT(model, tokenizer, batch_size)


def run_exp(args, session_runner):
    exp_start_time = time.time()
    queries_dict, runs = session_runner.run_sessions(args)
    print("finished running expriment time is:{} min".format((time.time() - exp_start_time) / 60))
    if args.log_lists:
        doc_searcher = collection_type_to_searcher(args.collection_type) if args.first_stage_ranker in ["ance",
                                                                                                     "tct"] else None
        output_as_initial_list(args.run_name,args.output_dir, runs, doc_searcher)
    if args.log_queries:
        queries_output_file = "{}/{}_queries.json".format(args.output_dir, args.run_name)
        output_queries_file(queries_output_file, queries_dict)

def collection_type_to_searcher(collection_type):
    if "cast" in collection_type:
        return SimpleSearcher.from_prebuilt_index("cast2019")
    #TODO: remove or_quac constant location
    if collection_type=="qrecc":
        return SimpleSearcher("/v/tomergur/convo/indexes/qrecc")
    if collection_type=="topiocqa":
        return SimpleSearcher("/v/tomergur/convo/indexes/topiocqa")
    if collection_type == "reddit":
        return SimpleSearcher("/v/tomergur/convo/indexes/reddit")
    return SimpleSearcher("/v/tomergur/convo/indexes/or_quac")

if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)
    args = parse_args()
    print(args)
    strategy = None
    if 'tpu' in args:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    args_output_path = "{}/{}_config.json".format(args.output_dir, args.run_name)

    with open(args_output_path, 'w') as f:
        json.dump(vars(args), f, indent=True)
    k1 = args.k1
    b = args.b
    count = args.count
    initial_lists = None
    if args.first_stage_ranker == "bm25":
        searcher = collection_type_to_searcher(args.collection_type)
        searcher.set_bm25(k1, b)
    if args.first_stage_ranker == "kld":
        searcher = collection_type_to_searcher(args.collection_type)
        searcher.set_qld(args.mu)
    elif args.first_stage_ranker == "cache":
        searcher = None
        initial_lists = RunsCache(args.first_stage_cache_path)

    #TODO: remove dense retrieval
    '''
    elif args.first_stage_ranker == "ance":
        searcher = SimpleDenseSearcher(args.index_path, 'castorini/ance-msmarco-passage')
    else:
        searcher = SimpleDenseSearcher(args.index_path, 'castorini/tct_colbert-msmarco')
    '''
    doc_fetcher = None
    if args.add_canonical_response and args.collection_type=="cast":
        if args.first_stage_ranker == "bm25":
            doc_fetcher = searcher
        else:
            doc_fetcher = collection_type_to_searcher(args.collection_type)
            doc_fetcher.set_bm25(k1, b)

    rewriters = []
    for rewriter_name in args.rewriters:
        rewriters.append(create_rewriter(rewriter_name, args))
    second_stage_rewriters = None
    if 'second_stage_rewriters' in args:
        second_stage_rewriters = []
        for rewriter_name in args.second_stage_rewriters:
            second_stage_rewriters.append(create_rewriter(rewriter_name, args))
    device = args.reranker_device if 'reranker_device' in args else None
    reranker = build_reranker(args, device=device,
                              strategy=strategy) if args.rerank else None
    hits_to_text_func=None
    if 'modify_documents_func' in args:
        with open(args.doc2q_path) as f:
            doc2q = json.load(f)
        if args.modify_documents_func=="doc2q_all":
            hits_to_text_func=lambda hits:modify_to_all_queries(doc2q,hits)
        elif args.modify_documents_func.startswith("doc2q_idx_"):
            q_idx=int(args.modify_documents_func.split("_")[-1])-1
            print("run modify idx:",q_idx)
            hits_to_text_func=lambda hits:modify_to_single_queries(doc2q,q_idx,hits)
        elif args.modify_documents_func=="doc2q_app_all":
                hits_to_text_func = lambda hits: modify_to_append_all_queries(doc2q, hits)

    pipeline = Pipeline(searcher, rewriters, count, reranker, second_stage_rewriters, initial_lists, args.log_queries,
                        hits_to_text_func)
    if args.collection_type=="cast":
        session_runner=CastSessionRunner(pipeline, doc_fetcher)
    elif args.collection_type=="topiocqa":
        session_runner = TopioCQASessionsRunner(pipeline, args.add_canonical_response)
    elif args.collection_type=="reddit":
        session_runner=RedditSessionRunner(pipeline)
    elif args.collection_type=="qrecc" or args.collection_type=="qrecc_cast":
        session_runner=QreccSessionRunner(pipeline,args.add_canonical_response)
    else:
        session_runner=ORQuacSessionRunner(pipeline,args.add_canonical_response)
    run_exp(args,session_runner)
