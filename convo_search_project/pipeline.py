import heapq
from pygaggle.rerank.base import Query, hits_to_texts
import datetime
import tensorflow as tf


class Pipeline():
    def __init__(self, searcher, rewriters, count, reranker=None, second_stage_rewriters=None,inital_lists=None, return_queries=False,hits_to_texts_func=None):
        self.searcher = searcher
        self.rewriters = rewriters
        self.count = count
        self.reranker = reranker
        self.second_stage_rewriters = second_stage_rewriters
        self.return_queries = return_queries
        self.cached_lists=inital_lists
        self.hits_to_texts=hits_to_texts_func if hits_to_texts_func is not None else hits_to_texts
        assert (len(rewriters)==0 or inital_lists is None)

    def rrf(self, runs, v=60):
        docs_scores = {}
        doc_results = {}
        for run in runs:
            for rank, doc in enumerate(run):
                docid = doc.docid
                doc_results[docid] = doc
                docs_scores[docid] = docs_scores.get(docid, 0) + 1. / (1 + rank + v)
        best_docs = heapq.nlargest(self.count, docs_scores.items(), key=lambda x: x[1])
        rrf_res = []
        for docid, score in best_docs:
            updated_doc = doc_results[docid]
            updated_doc.score = score
            rrf_res.append(updated_doc)
        return rrf_res

    def rerank(self, query, res_list):
        if len(res_list)==0:
            return res_list
        reranked = self.reranker.rerank(Query(query), self.hits_to_texts(res_list))
        reranked_scores = [r.score for r in reranked]
        # Reorder hits with reranker scores
        reranked = list(zip(res_list, reranked_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked_hits = []
        for r in reranked:
            doc = r[0]
            doc.score = r[1]
            reranked_hits.append(doc)
        return reranked_hits

    def retrieve(self, query, **ctx):
        if self.cached_lists:
            qid=ctx['qid']
            first_stage_lists=[self.cached_lists[qid][:self.count]]
            first_stage_queries=[query]
            queries_dict = {"query": query}
        else:
            first_stage_lists, first_stage_queries = self.first_stage_retrieval(ctx, query)
            queries_dict = {"query": query, "first_stage_rewrites": first_stage_queries}
        if not self.reranker:
            final_res_list = first_stage_lists[0] if len(first_stage_lists) == 1 else self.rrf(first_stage_lists)
            return final_res_list, queries_dict if self.return_queries else final_res_list

        reranked_lists, second_stage_queries = self.second_stage_retrieval(query, first_stage_lists,
                                                                           first_stage_queries, ctx)
        if len(second_stage_queries) > 0:
            queries_dict["second_stage_queries"] = second_stage_queries
        final_res_list = reranked_lists[0] if len(reranked_lists) == 1 else self.rrf(reranked_lists)
        return final_res_list, queries_dict if self.return_queries else final_res_list

    def second_stage_retrieval(self, query, first_stage_lists, first_stage_queries, ctx):
        second_stage_queries = []
        if self.second_stage_rewriters:
            # use early fusion
            initial_list = first_stage_lists[0] if len(first_stage_lists) == 1 else self.rrf(first_stage_lists)
            for rewriter in self.second_stage_rewriters:
                second_stage_rewrite = rewriter.rewrite(query, **ctx)
                if isinstance(second_stage_rewrite, list):
                    second_stage_queries += second_stage_rewrite
                else:
                    second_stage_queries.append(second_stage_rewrite)
            print("second stage query:", second_stage_queries)
            reranked_lists = [self.rerank(query, initial_list) for query in second_stage_queries]
        else:
            reranked_lists = [self.rerank(first_stage_query, run_res) for first_stage_query, run_res in
                              zip(first_stage_queries, first_stage_lists)]
        return reranked_lists, second_stage_queries

    def first_stage_retrieval(self, ctx, query):
        first_stage_queries = []
        for rewriter in self.rewriters:
            rewriter_res = rewriter.rewrite(query, **ctx)
            if isinstance(rewriter_res, list):
                first_stage_queries += rewriter_res
            else:
                first_stage_queries.append(rewriter_res)
        # handle the case where there are no rewriters
        if len(self.rewriters) == 0:
            first_stage_queries = [query]
        first_stage_lists = []
        print(first_stage_queries)
        for first_stage_query in first_stage_queries:
            run_res = self.searcher.search(first_stage_query, k=self.count)
            first_stage_lists.append(run_res)
        return first_stage_lists, first_stage_queries
