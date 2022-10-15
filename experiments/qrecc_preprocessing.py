import json
import os
QRECC_PATH="/v/tomergur/convo/qrecc/"
import pandas as pd
from pyserini.search import SimpleSearcher
import random
import tqdm
#qrel preprocess

def debug_index():
    searcher = SimpleSearcher("/v/tomergur/convo/indexes/qrecc_2/")
    print(searcher.num_docs)
    dups=[]
    docs=searcher.doc("https://v1.escapistmagazine.com/forums/read/7.860191-Ray-Rice-is-Out-of-the-NFL-On-His-Way-Out-of-EAs-Madden-15_p14")
    print(docs)
    for i in tqdm.tqdm(range(searcher.num_docs)):
        dups.append(searcher.doc(i).docid())
    dups=set(dups)
    print("num unique",len(dups))
def create_subset():
    input_dir = "/v/tomergur/convo/qrecc"
    #input_queries="qrecc-train.json"
    input_queries = "qrecc-train.json"
    #input_qrels="train_qrel.txt"
    input_qrels = "test_qrel.txt"
    dev_sids=0
    subset_size=300
    output_file="qrecc-test-subset.json"
    output_qrel="test_subset_qrel.txt"
    #output_train_file="qrecc-train-subset.json"
    #output_train_qrel="train_subset_qrel.txt"
    output_train_file="qrecc-test-left.json"
    output_train_qrel="test_left_subset_qrel.txt"
    qrels = pd.read_csv("{}/{}".format(input_dir, input_qrels), header=None, names=["qid", "Q0", "pid", "grade"],
                        dtype={"qid": str, "pid": str},
                        delimiter="\t")
    qrels=qrels.assign(sid=qrels.qid.map(lambda x:x.split("_")[0]))
    legal_sids=list(qrels.sid.unique())
    if dev_sids>0:
        legal_sids=[sid for sid in legal_sids if int(sid)%dev_sids==0]
    selected_sids=random.sample(legal_sids,subset_size)
    print(selected_sids)
    with open("{}/{}".format(input_dir,input_queries)) as f:
        dataset=json.load(f)
    sub_data=[data for data in dataset if str(data["Conversation_no"]) in selected_sids]
    remaining_data = [data for data in dataset if str(data["Conversation_no"]) not in selected_sids]
    print("sub length",len(sub_data))
    print("remaining_data",len(remaining_data))
    with open("{}/{}".format(input_dir,output_file),'w') as f:
        json.dump(sub_data,f,indent=True)
    with open("{}/{}".format(input_dir,output_train_file),'w') as f:
        json.dump(remaining_data,f,indent=True)
    sub_qrels=qrels[qrels.sid.isin(selected_sids)]
    sub_qrels[["qid", "Q0", "pid", "grade"]].to_csv("{}/{}".format(input_dir,output_qrel),index=False,header=None,sep="\t")
    train_qrels=qrels[~qrels.sid.isin(selected_sids)]
    train_qrels[["qid", "Q0", "pid", "grade"]].to_csv("{}/{}".format(input_dir,output_train_qrel),index=False,header=None,sep="\t")



def analyze_dial():
    with open("{}/{}".format(QRECC_PATH, "qrecc-train.json")) as f:
        dataset = json.load(f)
        for turn in dataset:
            if len(turn["Context"])!=2*(turn["Turn_no"]-1) or len(turn["Answer"])==0:
                print(len(turn["Context"]),2*(turn["Turn_no"]-1),turn["Conversation_no"],turn["Turn_no"])
                print(turn)

def prerpocess_passages():
    PASSAGES_PATH="/v/tomergur/convo/qrecc/collection-paragraph/"
    for dirpath, dirnames, filenames in os.walk(PASSAGES_PATH):
        for filename in filenames:
            print(dirpath,filename)
            with open("{}/{}".format(dirpath,filename)) as f:
                for line in f.readlines():
                    doc_json=json.loads(line)
                    print(doc_json["id"])
                    print(doc_json["contents"])



INPUT_DIR = "/v/tomergur/convo/qrecc"
#qrel preprocess
INPUT_QREL_FILE="train_qrel.txt"
#OUTPUT_QREL_FILE="or_qrels.txt"
ADD_REL=True
#add relevant passage to all quries_jsons
JSONS_DIR="/v/tomergur/convo/res/bm25_100/qrecc/train_manual_with_rel/"
def add_rel_to_queries():
    qrels=pd.read_csv("{}/{}".format(INPUT_DIR, INPUT_QREL_FILE), header=None, names=["qid", "Q0", "pid", "grade"], dtype={"qid": str, "pid": str},
                       delimiter="\t")
    searcher=SimpleSearcher("/v/tomergur/convo/indexes/qrecc/")
    for i,filename in enumerate(os.listdir(JSONS_DIR)):
        qid=filename.split(".")[0]
        print(qid,i,filename)
        with open("{}/{}".format(JSONS_DIR,filename)) as f:
            q_res=json.load(f)
        rel_list=qrels[qrels.qid==qid].pid.tolist()
        print(rel_list)
        rel=rel_list[0] if len(rel_list) >0 else None
        if len(rel_list) >1:
            print("two or more rels")
        if  len(rel_list) ==0:
            print("no rels")
        ret_ids=[doc["docid"] for doc in q_res]
        rel_ret=set(ret_ids).intersection(set(rel_list))
        if len(rel_ret)==0 and len(rel_list)>0 and ADD_REL:
            rel_doc = searcher.doc(rel)
            doc = {'docid': rel_doc.docid(), 'score': 0, 'rank': len(q_res), 'content': rel_doc.raw()}
            print(rel,rel_doc.docid(),ret_ids)
            print("rel added")
            q_res[-1] = doc

        for doc_res in q_res:
            try:
                content=json.loads(doc_res['content'])
                if type(content)==str:
                    print("parsed string")
                    continue
                doc_res['content'] = content['contents']
            except ValueError as e:
                if '{' in doc_res['content']:
                    #print(doc_res['content'])
                    #print(doc_res)
                    print("bad formatting")
        with open("{}/{}".format(JSONS_DIR, filename),'w') as f:
            json.dump(q_res,f)


def analyze_qrel():
    with open("{}/{}".format(QRECC_PATH,"qrecc-train_gt.json")) as f:
        dataset=json.load(f)
    print("number of questions",len(dataset))
    num_conv=set()
    num_gold=0
    gold_types={}
    gold_pass_hist={}
    qrel_data=[]
    total_gold=0
    for i,q in enumerate(dataset):
        num_conv.add(q["Conversation_no"])
        gold_passages=q.get("Truth_passages",[])
        if len(gold_passages)>0:
            qid="{}_{}".format(q["Conversation_no"],q["Turn_no"])
            for gold_passage in gold_passages:
                qrel_data.append({"qid":qid,"Q0": "Q0","docno":gold_passage,"conv_no":q["Conversation_no"],"label":1,"source":q.get("Conversation_source","unknown")})
            num_gold+=1
            total_gold+=len(gold_passages)
            gold_pass_hist[len(gold_passages)]=gold_pass_hist.get(len(gold_passages),0)+1
            conv_source=q.get("Conversation_source","unknown")
            gold_types[conv_source]=gold_types.get(conv_source,0)+1
    print("number of questions with gold passages",num_gold)
    print(list(gold_types.items()))
    print(list(gold_pass_hist.items()))
    data_df=pd.DataFrame(qrel_data)
    print(total_gold)
    print("total old conv:",len(data_df["conv_no"].unique()),len(num_conv))
    #data_df[["qid","Q0","docno","label"]].to_csv("{}/{}".format(QRECC_PATH, "test_qrel.txt"),header=None,index=False,sep="\t")
    data_df[["qid", "Q0", "docno", "label"]].to_csv("{}/{}".format(QRECC_PATH, "train_qrel.txt"), header=None,
                                                    index=False, sep="\t")


def analyze_qrecc_gold():
    qrel = pd.read_csv("{}/{}".format(QRECC_PATH, "test_qrel.txt"), header=None, names=["qid", "Q0", "pid", "grade"], dtype={"qid": str, "pid": str},
                       delimiter="\t")
    qrels=qrel[qrel.grade>0]
    hist = {k: set() for k in range(1, 14)}
    for qid,q_rows in qrels.groupby("qid"):
        print(qid)
        gold_passages=q_rows.pid.tolist()
        conversation_num, tid = qid.split("_")
        for gold_passage in gold_passages:
            num_session_rep=0
            for i in range(1,14):
                new_qid="{}_{}".format(conversation_num,i)
                new_qid_gold_passages=qrels[qrels.qid==new_qid].pid.tolist()
                if gold_passage in new_qid_gold_passages:
                    #print(qid,new_qid,gold_passage)
                    num_session_rep+=1
            hist[num_session_rep].add(gold_passage)
    total_gold=sum(len(x) for x in hist.values())
    for i,gold_set in hist.items():
        #print(i,len(gold_set),len(gold_set)/total_gold,sum(len(x) for y,x in hist.items() if y>=i)/total_gold)
        print(i,"&", len(gold_set), "&", round(sum(len(x) for y, x in hist.items() if y >= i) / total_gold,2),"\\ \hline")
    print("total gold",total_gold)




if __name__=="__main__":
    #analyze_qrel()
    #analyze_qrecc_gold()
    #prerpocess_passages()
    #analyze_dial()
    #add_rel_to_queries()
    create_subset()
    #debug_index()