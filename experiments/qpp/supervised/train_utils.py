from transformers import TFAutoModelForSequenceClassification, TFAutoModel, TFAutoModelForTokenClassification
from transformers.models.bert import BertConfig, TFBertForTokenClassification, TFBertForSequenceClassification
from experiments.qpp.supervised.groupwise_model import GroupwiseBert
from  experiments.qpp.supervised.bert_pl import BertPL
from  experiments.qpp.supervised.seq_qpp_model import SeqQPP
import tensorflow as tf
import keras
from dataclasses import dataclass

@dataclass
class ModelArguments:
    model_type:str ="bert_qpp"
    model_name_or_path: str = "bert_base_uncased"
    group_model_name_or_path: str = None
    chunk_size: int =2
    use_bert_pl: bool = False
    from_pt: bool = False
    groupwise_model: bool = False
    group_agg_func: str = None
    output_mode : str = None
    use_mse: bool = False

def create_model(model_args):
    model_name=model_args.model_name_or_path
    if model_args.model_type=="seqQPP":
        model = TFAutoModel.from_pretrained(model_name, from_pt=model_args.from_pt)
        return SeqQPP(model)
    if hasattr(model_args, "use_bert_pl" ) and model_args.use_bert_pl:
        model = TFAutoModel.from_pretrained(model_name, from_pt=model_args.from_pt)
        return BertPL(model,model_args.chunk_size)

    num_classes = 1 if model_args.use_mse else 2
    if model_args.groupwise_model:
        model = TFAutoModel.from_pretrained(model_name, from_pt=model_args.from_pt)
        if model_args.group_model_name_or_path:
            group_model = TFBertForSequenceClassification.from_pretrained(
                model_args.group_model_name_or_path, from_pt=True, num_hidden_layers=4,
                num_labels=num_classes) if "seq" in model_args.output_mode else TFBertForTokenClassification.from_pretrained(
                model_args.group_model_name_or_path, from_pt=True, num_hidden_layers=4, num_labels=num_classes)
        else:
            group_conf = BertConfig(num_hidden_layers=4, num_labels=num_classes)
            group_model = TFBertForSequenceClassification(
                group_conf) if "seq" in model_args.output_mode  else TFBertForTokenClassification(group_conf)
        return GroupwiseBert(model, group_model, model_args.group_agg_func, model_args.output_mode)

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=model_args.from_pt,
                                                                 num_labels=num_classes)
    return model

def ce_loss(y_true, y_pred):
    scores = tf.nn.log_softmax(y_pred)
    y_true=tf.reshape(y_true,[-1,1])
    non_rel_prob = tf.ones_like(y_true, dtype=tf.float32) - y_true
    probs=tf.concat([non_rel_prob, y_true], axis=1)
    #tf.print(probs)
    #tf.print("loss:", tf.shape(y_pred),tf.shape(probs),tf.shape(y_true))
    loss = -1 * (tf.math.multiply(scores,probs))
    loss = tf.reduce_sum(loss, axis=-1)
    return loss

class CheckpointTransformerModel(keras.callbacks.Callback):
    def __init__(self,model_path,tokenizer):
        super(CheckpointTransformerModel, self).__init__()
        self.model_path=model_path
        self.tokenizer=tokenizer
    def on_epoch_end(self, epoch,logs=None):
        cur_path=self.model_path.format(epoch+1)
        #print("cur path",cur_path)
        self.model.save_pretrained(cur_path)
        self.tokenizer.save_pretrained(cur_path)
