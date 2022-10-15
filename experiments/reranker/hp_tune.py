import argparse
import pandas as pd
import os
METRIC='recip_rank'
#METRIC="ndcg_cut_3"
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",default="/lv_local/home/tomergur/convo_search_project/data/eval/rerank_dpr/")
    parser.add_argument("--collection",default="qrecc_dev")
    parser.add_argument("--run_name_filter",default=None)
    parser.add_argument("--metric",default=METRIC)
    args = parser.parse_args()
    input_path="{}/{}".format(args.input_path,args.collection)
    metric=args.metric
    run_name_filter=args.run_name_filter
    runs_df=[]
    for filename in os.listdir(input_path):
        if run_name_filter is not None and run_name_filter not in filename:
            continue
        f = os.path.join(input_path, filename)
        df=pd.read_csv(f, header=None, names=["metric", "qid", "value"], delimiter="\t")
        df.metric = df.metric.str.strip()
        df=df[df.metric==metric]
        #print(f)
        all_res=df.tail(1)
        #print(all_res)
        #print(filename[:-4])
        runs_df.append({"run_name":filename[:-4],"metric":all_res.value.iloc[0]})
    res_df=pd.DataFrame(runs_df)
    print(res_df.sort_values("run_name"))
    print(res_df[res_df.metric==res_df.metric.max()])
