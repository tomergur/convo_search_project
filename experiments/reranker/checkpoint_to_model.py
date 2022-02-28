import tensorflow as tf
import argparse
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args=parser.parse_args()
    model_name_or_path="castorini/monobert-large-msmarco-finetune-only"
    input_path=args.input_path
    output_path=args.output_path
    checkpoints_dirs=tf.io.gfile.glob(input_path)
    print(checkpoints_dirs)
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, from_pt=True, num_labels=2)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for checkpoint_dir in checkpoints_dirs:
        checkpoint_num=checkpoint_dir.split("checkpoint")[-1]
        print(checkpoint_num)
        model.load_weights(checkpoint_dir+"/")
        output_dir="{}/checkpoint{}/".format(output_path,checkpoint_num)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        tokenizer.save_pretrained(output_dir)
        model.save_pretrained(output_dir)


