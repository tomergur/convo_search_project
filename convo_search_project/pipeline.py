import heapq
from pygaggle.rerank.base import Query, Reranker, hits_to_texts


class Pipeline():
    def __init__(self, searcher, rewriters, count, reranker=None, second_stage_rewriter=None, return_queries=False):
        self.searcher = searcher
        self.rewriters = rewriters
        self.count = count
        self.reranker = reranker
        self.second_stage_rewriter = second_stage_rewriter
        self.return_queries = return_queries

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
        # print([(x.docid,x.score) for x in res_list[:10]])
        reranked = self.reranker.rerank(Query(query), hits_to_texts(res_list))
        reranked_scores = [r.score for r in reranked]
        # print([(x.docid,y.metadata['docid']) for x,y in list(zip(res_list, reranked))[:10]])
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
        queries_dict = {"query": query, "first_stage_rewrites": first_stage_queries}
        first_stage_lists = []
        print(first_stage_queries)
        for first_stage_query in first_stage_queries:
            run_res = self.searcher.search(first_stage_query, k=self.count)
            first_stage_lists.append(run_res)

        if not self.reranker:
            final_res_list = first_stage_lists[0] if len(first_stage_lists) == 1 else self.rrf(first_stage_lists)
            return final_res_list, queries_dict if self.return_queries else final_res_list

        # use early fusion
        if self.second_stage_rewriter:
            early_fusion_list = self.rrf(first_stage_lists)
            second_stage_query = self.second_stage_rewriter.rewrite(query, **ctx)
            assert not isinstance(second_stage_query, list)
            print("second stage query:", second_stage_query)
            return self.rerank(second_stage_query, early_fusion_list)

        reranked_lists = [self.rerank(first_stage_query, run_res) for first_stage_query, run_res in
                          zip(first_stage_queries, first_stage_lists)]
        final_res_list = reranked_lists[0] if len(reranked_lists) == 1 else self.rrf(reranked_lists)
        return final_res_list, queries_dict if self.return_queries else final_res_list
