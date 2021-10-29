import json
import tqdm
import argparse
import tensorflow as tf
COL='2019'
METHOD='prev_first_turn_base'
model_name="castorini/doc2query-t5-base-msmarco"
from transformers import T5Tokenizer,TFT5ForConditionalGeneration
NUM_RETURNED_QUERIES=20

'''
def generate_queries(src_text,tokenizer,model):
    input_ids = tokenizer(
        src_text,max_length=512,truncation=True, return_tensors="tf", add_special_tokens=True)

    output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=NUM_RETURNED_QUERIES)
    # Decode output
    queries_genereated = []
    for i in range(NUM_RETURNED_QUERIES):
        query_rewrite = tokenizer.decode(
            output_ids[i, 0:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True)
        queries_genereated.append(query_rewrite)
    return queries_genereated
'''

def generate_queries(src_texts,tokenizer,model):
    input_ids = tokenizer(
        src_texts,max_length=512, padding="max_length",truncation=True, return_tensors="tf", add_special_tokens=True)


    output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],
        max_length=64,
        do_sample=True,
        top_k=10,
        num_return_sequences=NUM_RETURNED_QUERIES)
    # Decode output
    docs_res=[]
    for j in range(len(src_texts)):
        queries_genereated = []
        for i in range(NUM_RETURNED_QUERIES):
            query_rewrite = tokenizer.decode(
            output_ids[j*NUM_RETURNED_QUERIES+i, 0:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True)
            queries_genereated.append(query_rewrite)
        docs_res.append(queries_genereated)

    return docs_res


def main(args):
    col = args.collection
    method = args.method
    LISTS_PATH = "./data/res/{}_{}_lists.json".format(col, method)
    OUTPUT_PATH = "./data/res/{}_{}_doc2q.json".format(col, method)
    res = {}
    tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True)
    model = TFT5ForConditionalGeneration.from_pretrained(model_name, from_pt=True)
    with open(LISTS_PATH) as f:
        res_lists = json.load(f)
    for qid, res_list in list(res_lists.items()):
        print(qid)
        q_res = {}
        #
        for idx in tqdm.tqdm(range(0, len(res_list), args.batch_size)):
            batch_list = res_list[idx:idx + args.batch_size]
            doc_q = generate_queries([doc['content'] for doc in batch_list], tokenizer, model)
            for doc, q in zip(batch_list, doc_q):
                q_res[doc['docid']] = q
        res[qid] = q_res
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--collection",default="2019")
    parser.add_argument("--method",default="prev_first_turn_base")
    parser.add_argument("--tpu",default=None)
    args=parser.parse_args()
    if args.tpu is not None:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
        with strategy.scope():
            main(args)
    else:
        main(args)


