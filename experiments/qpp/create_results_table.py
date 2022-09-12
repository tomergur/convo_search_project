import pandas as pd

RES_DIR="rerank_kld_100"
COL="topiocqa"
#"baseline",
METHODS=["turn_number","q_len","avg_idf","avg_scq","avg_var","max_idf","max_scq","max_var","clarity","WIG","NQC","bert_qpp_or_quac","bert_qpp_topiocqa"]
METHODS=["turn_number","q_len","avg_idf","avg_scq","avg_var","max_idf","max_scq","avg_var","clarity_norm","WIG_norm","NQC_norm","bert_qpp","bert_qpp_or_quac"]
METHODS2=["avg_idf","avg_scq","avg_var","max_idf","max_scq","max_var","clarity","WIG","NQC"]
#METHODS2=["avg_idf","avg_scq","avg_var","max_idf","max_scq","avg_var","clarity_norm","WIG_norm","NQC_norm"]
#METHODS2=["avg_idf","avg_scq","avg_var","max_idf","max_scq","avg_var"]
VARIANTS=["","ref_hist_","ref_hist_decay_","ref_rewrites_"]
VARIANTS=["","ref_hist_","ref_rewrites_"]
VAR_NAMES={"":"baseline","ref_hist_":"history ref. queries","ref_rewrites_":"other rewrites as ref queries",
           "ref_hist_decay_": "history ref queries with decay turns similarity"}
REWRITE_METHODS=['raw','t5','all','hqe','quretec','manual']
REWRITE_METHODS=['t5','all','hqe','quretec']
#APPENDIX="_30"
APPENDIX=""
QPP_DIR="/lv_local/home/tomergur/convo_search_project/data/qpp/topic_comp/"
METRIC="map_cut_1000"
METRIC="recip_rank"
RES_TYPE="corr"
#RES_TYPE="threshold"
def single_results_df():
    res=None
    for method in METHODS:
        method_res_path="{}/{}/{}/{}_{}_{}{}.csv".format(QPP_DIR,RES_DIR,COL,RES_TYPE,method,METRIC,APPENDIX)
        method_res=pd.read_csv(method_res_path)
        res=method_res if res is None else res.append(method_res)
    output_path="{}/{}/{}/{}_res_{}.csv".format(QPP_DIR,RES_DIR,COL,RES_TYPE,METRIC)
    res=res.assign(predictor=res.predictor.str.replace("_"," "))
    res.to_csv(output_path,index=False)
    print(res)

def comparing_variants_df(rewrite_method):
    res=[]
    for method in METHODS2:
        row = {"predictor": method}
        for variant in VARIANTS:
            method_var_name=variant+method
            method_res_path = "{}/{}/{}/{}_{}_{}.csv".format(QPP_DIR, RES_DIR, COL,RES_TYPE, method_var_name, METRIC,APPENDIX)
            method_res = pd.read_csv(method_res_path)
            method_val=method_res[rewrite_method].to_list()[0]
            row[VAR_NAMES[variant]]=method_val
        res.append(row)
    res_df=pd.DataFrame(res)
    output_path="{}/{}/{}/comp_{}_{}_res_{}.csv".format(QPP_DIR,RES_DIR,COL,rewrite_method,RES_TYPE,METRIC)
    res_df.to_csv(output_path,index=False)
    print(rewrite_method)
    print(res_df)











if __name__=="__main__":
    single_results_df()
    for rewrite_method in REWRITE_METHODS:
        comparing_variants_df(rewrite_method)


