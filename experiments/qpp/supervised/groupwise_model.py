import tensorflow as tf
from transformers import TFAutoModel, TFAutoModelForTokenClassification, TFAutoModelForSequenceClassification

import h5py
from tensorflow.python.keras.saving import hdf5_format


class GroupwiseBert(tf.keras.Model):
    def __init__(self, text_bert, group_bert, agg_func=None, output_mode=None,max_seq_length=None):
        super(GroupwiseBert, self).__init__()
        self.text_bert = text_bert
        self.group_bert = group_bert
        self.output_mode = output_mode
        self.max_seq_length=max_seq_length
        if agg_func == "max":
            self.agg_func = tf.reduce_max
        elif agg_func == "mean":
            self.agg_func = tf.reduce_mean
        elif agg_func == "first":
            self.agg_func = lambda x: x[:, 0]
        elif agg_func == "last":
            self.agg_func = lambda x: x[:, -1] if tf.shape(x)[1]>0 else x[0]
        else:
            self.agg_func = None
        # self.group_bert.layers[0].embeddings.word_embeddings.trainable=False

    @staticmethod
    def from_pretrained(model_path, group_agg_func=None, output_mode=None):

        # group_model = TFAutoModelForTokenClassification.from_pretrained(group_model_path)
        # "groupwise_bert"
        text_model_path=model_path+"/text_embed/"
        group_model_path=model_path+"/group_model/"
        text_model = TFAutoModel.from_pretrained(text_model_path)

        with tf.name_scope("groupwise_bert") as scope:
            group_model = TFAutoModelForSequenceClassification.from_pretrained(
                group_model_path) if "seq" in output_mode else TFAutoModelForTokenClassification.from_pretrained(
                group_model_path)
        '''
        group_model = TFAutoModelForSequenceClassification.from_pretrained(
                group_model_path) if "seq" in output_mode else TFAutoModelForTokenClassification.from_pretrained(
                group_model_path)
         '''
        return GroupwiseBert(text_model, group_model, group_agg_func, output_mode=output_mode)

    def limit_seq_length_online_output(self,text_emb,training):
        #assuming a single sequence
        seq_length = tf.shape(text_emb)[1]
        #tf.math.maximum(0,i-self.max_seq_length)
        short_seq=tf.map_fn(lambda i: tf.pad(text_emb[0,0:i+1,:],paddings=[[0,self.max_seq_length-i],[0,0]]),tf.range(self.max_seq_length),fn_output_signature=tf.float32)
        sequences=tf.map_fn(lambda i: text_emb[0,i:i+1+self.max_seq_length,:],tf.range(seq_length-self.max_seq_length),fn_output_signature=tf.float32)
        seq_text_emb=tf.concat([short_seq,sequences],0)
        #tf.print("seq text emb",tf.shape(seq_text_emb))
        att_mask = tf.ones((seq_length, self.max_seq_length+1))
        att_mask = tf.linalg.band_part(att_mask, -1, -0)
        group_inputs = {'inputs_embeds': seq_text_emb, 'attention_mask': att_mask}
        res = self.group_bert(group_inputs, training=training)
        if self.output_mode == "online_seq":
            return res.logits
        tokens_indices=tf.concat([tf.range(self.max_seq_length),tf.repeat(self.max_seq_length,repeats=[seq_length-self.max_seq_length], axis=0)],axis=0)
        token_res=tf.gather(res.logits, indices=tokens_indices, axis=1, batch_dims=1)
        return token_res

    def online_output(self, text_emb, training):
        # seq_length=text_emb.shape[1]
        seq_length = tf.shape(text_emb)[1]
        rep_text_emb = tf.repeat(text_emb, repeats=[seq_length], axis=0)
        att_mask = tf.ones((seq_length, seq_length))
        att_mask = tf.linalg.band_part(att_mask, -1, -0)
        group_inputs = {'inputs_embeds': rep_text_emb, 'attention_mask': att_mask}
        res = self.group_bert(group_inputs, training=training)
        if self.output_mode == "online_seq":
            return res.logits
        token_res=tf.gather(res.logits, indices=tf.range(seq_length), axis=1, batch_dims=1)
        return token_res


    def call(self, inputs, training=False):
        #tf.print(tf.shape(inputs['input_ids']))
        input_shape=tf.shape(inputs['input_ids'])
        num_seq = input_shape[0]
        seq_length = input_shape[1]
        entry_length=input_shape[2]
        inputs = {k: tf.reshape(v, [-1, entry_length]) for k, v in inputs.items()}
        #tf.print(tf.shape(inputs['input_ids']))
        bert_res = self.text_bert(**inputs, training=training)
        # text_emb=tf.expand_dims(bert_res.last_hidden_state[:,0,:],0)
        #text_emb = tf.expand_dims(bert_res.pooler_output, 0)
        #tf.print(tf.shape(text_emb))
        text_emb=tf.reshape(bert_res.pooler_output,[num_seq,seq_length,-1])
        #tf.print(tf.shape(text_emb))
        if self.output_mode and "online" in self.output_mode:
            if self.max_seq_length is not None:
                return self.limit_seq_length_online_output(text_emb, training)
            return self.online_output(text_emb, training)
        group_inputs = {'inputs_embeds': text_emb}
        res = self.group_bert(group_inputs, training=training)
        # print(tf.shape(res.logits))
        # group_res=tf.squeeze(res.logits,0)
        group_res = res.logits
        if self.agg_func:
            return self.agg_func(group_res)
        return group_res

    def save_pretrained(self, output_path):
        text_embed_path = "{}/text_embed".format(output_path)
        group_path = "{}/group_model".format(output_path)
        self.text_bert.save_pretrained(text_embed_path)
        self.group_bert.save_pretrained(group_path)
