from transformers import TFAutoModelForSequenceClassification, TFAutoModel, TFAutoModelForTokenClassification
from transformers.models.bert import BertConfig, TFBertForTokenClassification, TFBertForSequenceClassification
from experiments.qpp.supervised.groupwise_model import GroupwiseBert
import tensorflow as tf


def create_model(model_name, data_args):
    num_classes = 1 if data_args.use_mse else 2
    if data_args.groupwise_model:
        model = TFAutoModel.from_pretrained(model_name, from_pt=data_args.from_pt)
        if data_args.group_model_name_or_path:
            group_model = TFBertForSequenceClassification.from_pretrained(data_args.group_model_name_or_path,
                                                                          from_pt=True, num_hidden_layers=4,
                                                                          num_labels=num_classes)
            group_model = TFBertForSequenceClassification(
                data_args.group_model_name_or_path, from_pt=True, num_hidden_layers=4,
                num_labels=num_classes) if data_args.output_mode == "online_seq" else TFBertForTokenClassification(
                data_args.group_model_name_or_path, from_pt=True, num_hidden_layers=4, num_labels=num_classes)
        else:
            group_conf = BertConfig(num_hidden_layers=4, num_labels=num_classes)
            group_model = TFBertForSequenceClassification(
                group_conf) if data_args.output_mode == "online_seq" else TFBertForTokenClassification(group_conf)
        return GroupwiseBert(model, group_model, data_args.group_agg_func, data_args.output_mode)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=data_args.from_pt,
                                                                 num_labels=num_classes)
    return model


def ce_loss(y_true, y_pred):
    scores = tf.nn.log_softmax(y_pred)
    non_rel_prob = tf.ones_like(y_true, dtype=tf.float32) - y_true
    loss = -1 * (tf.math.multiply(scores, tf.concat([non_rel_prob, y_true], axis=1)))
    loss = tf.reduce_sum(loss, axis=-1)
    return loss
