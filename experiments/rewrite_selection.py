import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import HfArgumentParser, TFTrainingArguments
from dataclasses import dataclass, field
import tensorflow_ranking as tfr
import json
import os
import pandas as pd
import itertools
import time

from experiments.ltr.ltr_utils import calc_metric, write_outputs,TARGET_COLUMNS,create_dataset,load_or_create_folds
def create_model(model_name):
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    for layer in model.layers[:-1]:
        print(layer)
        layer.trainable = False
    print(model.summary())
    return model

@dataclass
class DataArguments:

    dataset_name: str = "dialog_rep_dataset"
    input_dir:str= "/lv_local/home/tomergur/convo_search_project/data/ltr"
    base_dir:str="/lv_local/home/tomergur/convo_search_project/data"
    folds_file_name:str="folds_10.json"
    model_name: str="bert-base-uncased"
    eval_dir_format: str ="{}/eval/rerank_3000/cast{}/"
    collection:str="both"
    folds_num:int=10
    checkpoint_dir: str =None
    checkpoint_step: int = 0
    from_pt:bool= False

def create_rewrites_dataset(data_args,input_lists,sids,tokenizer,training):
    start_time=time.time()
    if data_args.collection=="both":
        ds_19 =create_rewrites_dataset_collection(data_args,input_lists,sids,tokenizer,training,"19")
        ds_20 = create_rewrites_dataset_collection(data_args, input_lists, sids, tokenizer,training, "20")
        res_ds=ds_19.concatenate(ds_20)
    else:
        res_ds=create_rewrites_dataset_collection(data_args,input_lists,sids,tokenizer,training,data_args.collection)
    print("dataset creation time:", time.time() - start_time)
    if training:
        return res_ds.shuffle(1000)
    return res_ds

def create_rewrites_dataset_collection(data_args,input_lists,sids,tokenizer,training,collection):
    eval_dir=data_args.eval_dir_format.format(data_args.base_dir,collection)
    queries_dicts={}
    metric_dicts={}
    for input_list in input_lists:
        qrel=pd.read_csv("{}/{}.txt".format(eval_dir,input_list),header=None, names=["metric","qid", "value"],delimiter="\t")
        qrel=qrel[~(qrel.qid=="all")]
        #print(len(qrel),qrel)
        qrel=qrel.assign(sid=qrel.qid.map(lambda x:int(x.split("_")[0])))
        qrel=qrel[qrel.sid.isin(sids)]
        #print(len(qrel),qrel)
        metric_dict=qrel[qrel.metric.str.startswith("ndcg_cut_3")].set_index("qid").value.to_dict()
        metric_dicts[input_list]=metric_dict
        with open("/{}/res/rerank_3000/cast{}/{}_queries.json".format(data_args.base_dir,collection,input_list)) as f:
            rewrite_json=json.load(f)
            queries_dicts[input_list]={qid:val["second_stage_queries"][0] for qid,val in rewrite_json.items()}
    qids=metric_dicts[input_lists[0]].keys()
    diag=[]
    rewrite=[]
    class_label=[]
    res_qids=[]
    methods=[]
    for qid in qids:
        qid_res=[metric_dicts[input_list][qid] for input_list in input_lists]
        best_val=max(qid_res)
        labels=[1 if val==best_val else 0 for val in qid_res]
        if not training:
            diag+=[queries_dicts["all"][qid]]*len(labels)
            rewrite+=[queries_dicts[input_list][qid] for input_list in input_lists]
            class_label+=labels
            res_qids+=[qid]*len(labels)
            methods+=[input_list for input_list in input_lists]
            continue
        for i,j in itertools.combinations(range(len(input_lists)), 2):
            if labels[i]==labels[j]:
                continue
            diag.append(queries_dicts["all"][qid])
            rewrite.append(queries_dicts[input_lists[i]][qid])
            class_label.append(labels[i])
            res_qids.append(qid)
            methods.append(input_lists[i])
            diag.append(queries_dicts["all"][qid])
            rewrite.append(queries_dicts[input_lists[j]][qid])
            class_label.append(labels[j])
            methods.append(input_lists[j])
            res_qids.append(qid)
    ret=tokenizer(diag,rewrite, max_length=512,truncation=True, padding="max_length", return_token_type_ids=True, return_tensors='tf')
    tf_ds = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": ret['input_ids'], "attention_mask": ret['attention_mask'],
         "token_type_ids": ret['token_type_ids'],"qids":res_qids,"methods":methods},class_label))
    return tf_ds

def output_eval_run(test_df,data_args,input_lists,run_name):
    output_eval_run_collection(test_df, data_args, input_lists,run_name, '19')
    output_eval_run_collection(test_df, data_args, input_lists,run_name, '20')

def output_eval_run_collection(test_df,data_args,input_lists,run_name,collection):
    eval_dir = data_args.eval_dir_format.format(data_args.base_dir, collection)
    eval_dicts={}
    res=None
    test_df=test_df.assign(sid=test_df.qid.map(lambda x:int(x.split("_")[0])))
    test_df = test_df[test_df.sid <= 80] if collection == '19' else test_df[test_df.sid > 80]
    for input_list in input_lists:
        qrel = pd.read_csv("{}/{}.txt".format(eval_dir, input_list), header=None, names=["metric", "qid", "value"],
                           delimiter="\t")
        eval_dicts[input_list] = qrel
    for i,row in test_df.iterrows():
        qid=row["qid"]
        best_method=eval_dicts[row["method"]]
        qid_res=best_method[best_method.qid==qid]
        res=res.append(qid_res) if res is not None else qid_res
        all_data=[]
    for metric in res.metric.unique():
        val = round(res[res.metric == metric].value.mean(), 4)
        all_data.append({"qid": "all", "metric": metric, "value": val})
    res = res.append(pd.DataFrame(all_data))
    output_path = "{}/{}.txt".format(eval_dir,run_name)
    res.to_csv(output_path, index=False, header=False, sep='\t')



INPUT_LISTS=["t5","all","t5_all_v3"]
if __name__=="__main__":
    parser = HfArgumentParser((DataArguments, TFTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    input_lists=INPUT_LISTS
    # Create a description of the features.
    model_name = data_args.model_name
    '''
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    '''
    test_res=None
    with training_args.strategy.scope():
        run_name = training_args.run_name
        folds = load_or_create_folds(data_args, None)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for i,fold in enumerate(folds,start=1):
            print("fold:", i)
            train_sids = fold["train"]
            valid_sids = fold["valid"]
            test_sids = fold["test"]
            model = create_model(model_name)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            '''
                  loss = tfr.keras.losses.get(
                      loss=tfr.keras.losses.RankingLossKey.PAIRWISE_LOGISTIC_LOSS,
                      lambda_weight=tfr.losses.create_ndcg_lambda_weight(), ragged=True)
                  '''
            eval_metrics = [tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=False),tfr.keras.metrics.get(key="ndcg", name="metric/ndcg@3", ragged=False, topn=3)]
            eval_metrics=['accuracy']
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=training_args.learning_rate), loss=loss,
                      metrics=eval_metrics)
            ds_train = create_rewrites_dataset(data_args,input_lists,train_sids,tokenizer,True)
            ds_valid = create_rewrites_dataset(data_args,input_lists,valid_sids,tokenizer,True)
            callbacks = []
            history = model.fit(ds_train.batch(training_args.train_batch_size), epochs=int(training_args.num_train_epochs), verbose=1,
                                    callbacks=callbacks,validation_data=ds_valid.batch(training_args.eval_batch_size))
            print(history)
            #tokenizer.save_pretrained(training_args.output_dir)
            #model.save_pretrained(training_args.output_dir)

            #do eval
            ds_test=create_rewrites_dataset(data_args,input_lists,test_sids,tokenizer,False)
            output=model.predict(ds_test.batch(training_args.eval_batch_size))
            scores=tf.nn.softmax(output["logits"])[:,1]
            res_qids,methods=[],[]
            for x,y in ds_test:
                res_qids.append(x["qids"].numpy().decode())
                methods.append(x["methods"].numpy().decode())
            test_df=pd.DataFrame({"qid":res_qids,"method":methods,"scores":scores})
            test_df=test_df.assign(ranks=test_df.groupby("qid").scores.rank(ascending=False))
            selected_test=test_df[test_df.ranks==1]
            test_res=test_res.append(selected_test) if test_res is not None else selected_test
        output_eval_run(test_res, data_args, input_lists,run_name)








