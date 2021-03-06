from pygaggle.rerank.base import Text
MAX_Q=5
def modify_to_all_queries(doc_q,hits):
    texts = []
    for i,hit in enumerate(hits):
        t =  "[SEP]".join(doc_q[hit.docid][:MAX_Q])
        metadata = {'raw': hit.raw, 'docid': hit.docid}
        texts.append(Text(t, metadata, hit.score))
    return texts

def modify_to_append_all_queries(doc_q,hits):
    texts = []
    for i,hit in enumerate(hits):
        t =  "[SEP]".join([hit.raw]+doc_q[hit.docid][:MAX_Q])
        metadata = {'raw': hit.raw, 'docid': hit.docid}
        texts.append(Text(t, metadata, hit.score))
    return texts

def modify_to_single_queries(doc_q,idx,hits):
    texts = []
    for i,hit in enumerate(hits):
        t = doc_q[hit.docid][idx]
        metadata = {'raw': hit.raw, 'docid': hit.docid}
        texts.append(Text(t, metadata, hit.score))
    return texts