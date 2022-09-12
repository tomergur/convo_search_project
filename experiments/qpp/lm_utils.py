from .lm import UnigramLM
from .dir_smoothed_lm import DirSmoothLM

def create_doc_lm(docid,index_reader):
    doc_tfs = index_reader.get_document_vector(docid)
    doc_tfs={t:v for t,v in doc_tfs.items() if index_reader.get_term_counts(t,analyzer=None)[0]>0}
    doc_len=sum(doc_tfs.values())
    doc_lm={term:tf/doc_len for term,tf in doc_tfs.items()}
    return UnigramLM(doc_lm)

def create_doc_dir_smooth_lm(docid,index_reader,col_lm,mu=1000):
    doc_tfs = index_reader.get_document_vector(docid)
    doc_len=sum(doc_tfs.values())
    return DirSmoothLM( mu, doc_tfs, doc_len, col_lm)

def create_q_lm(query,index_reader):
    q_tf={}
    for term in query.split():
        analyzed = index_reader.analyze(term)
        if len(analyzed)==0:
            continue
        q_tf[analyzed[0]]=1+q_tf.get(analyzed[0],0)
    q_tf={t:v for t,v in q_tf.items() if index_reader.get_term_counts(t,analyzer=None)[0]>0 }
    query_len=sum(q_tf.values())
    q_lm={t:tf/query_len for t,tf in q_tf.items()}
    return UnigramLM(q_lm)