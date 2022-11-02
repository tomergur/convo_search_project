import tensorflow as tf
from transformers import TFAutoModel,TFAutoModelForTokenClassification,TFAutoModelForSequenceClassification

import h5py
from tensorflow.python.keras.saving import hdf5_format
class GroupwiseBert(tf.keras.Model):
    def __init__(self,text_bert,group_bert,agg_func=None,output_mode=None):
        super(GroupwiseBert, self).__init__()
        self.text_bert=text_bert
        self.group_bert=group_bert
        self.output_mode=output_mode
        if agg_func=="max":
            self.agg_func=tf.reduce_max
        elif agg_func=="mean":
            self.agg_func=tf.reduce_mean
        elif agg_func=="first":
            self.agg_func=lambda x:x[0,0]
        elif agg_func=="last":
            self.agg_func=lambda x:x[0,-1]
        else:
            self.agg_func=None
        #self.group_bert.layers[0].embeddings.word_embeddings.trainable=False

    @staticmethod
    def from_pretrained(text_model_path,group_model_path,group_agg_func=None,output_mode=None):
        text_model=TFAutoModel.from_pretrained(text_model_path)
        #group_model = TFAutoModelForTokenClassification.from_pretrained(group_model_path)
        # "groupwise_bert"

        with tf.name_scope("groupwise_bert") as scope:
            group_model=TFAutoModelForSequenceClassification.from_pretrained(group_model_path)

        return GroupwiseBert(text_model,group_model,group_agg_func,output_mode=output_mode)

    def online_output(self,text_emb,training):
        #seq_length=text_emb.shape[1]
        seq_length=tf.shape(text_emb)[1]
        rep_text_emb=tf.repeat(text_emb,repeats=[seq_length],axis=0)
        att_mask = tf.ones((seq_length, seq_length))
        att_mask=tf.linalg.band_part(att_mask, -1, -0)
        group_inputs={'inputs_embeds':rep_text_emb,'attention_mask':att_mask}
        res=self.group_bert(group_inputs,training=training)
        #tf.gather(res.logits, indices=range(seq_length), axis=1, batch_dims=1)
        return res.logits



    def call(self,inputs,training=False):
        bert_res=self.text_bert(**inputs,training=training)
        #text_emb=tf.expand_dims(bert_res.last_hidden_state[:,0,:],0)
        text_emb = tf.expand_dims(bert_res.pooler_output, 0)
        if self.output_mode=="online":
            return self.online_output(text_emb,training)
        group_inputs={'inputs_embeds':text_emb}
        res=self.group_bert(group_inputs,training=training)
        group_res=tf.squeeze(res.logits,0)
        if self.agg_func:
            return self.agg_func(group_res)
        return group_res

    def save_pretrained(self, text_embed_path,group_path):
        self.text_bert.save_pretrained(text_embed_path)
        self.group_bert.save_pretrained(group_path)