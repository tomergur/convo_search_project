import tensorflow as tf
from transformers import TFAutoModel,TFAutoModelForTokenClassification

import h5py
from tensorflow.python.keras.saving import hdf5_format
class GroupwiseBert(tf.keras.Model):
    def __init__(self,text_bert,group_bert):
        super(GroupwiseBert, self).__init__()
        self.text_bert=text_bert
        self.group_bert=group_bert
        #self.group_bert.layers[0].embeddings.word_embeddings.trainable=False

    @staticmethod
    def from_pretrained(text_model_path,group_model_path):
        text_model=TFAutoModel.from_pretrained(text_model_path)
        #group_model = TFAutoModelForTokenClassification.from_pretrained(group_model_path)
        # "groupwise_bert"

        with tf.name_scope("groupwise_bert") as scope:
            group_model=TFAutoModelForTokenClassification.from_pretrained(group_model_path)

        return GroupwiseBert(text_model,group_model)

    def call(self,inputs,training=False):
        bert_res=self.text_bert(**inputs,training=training)
        #text_emb=tf.expand_dims(bert_res.last_hidden_state[:,0,:],0)
        text_emb = tf.expand_dims(bert_res.pooler_output, 0)
        group_inputs={'inputs_embeds':text_emb}
        res=self.group_bert(group_inputs,training=training)
        #res = self.group_bert(inputs_embeds=text_emb, training=training)
        group_res=tf.squeeze(res.logits,0)
        return group_res

    def save_pretrained(self, text_embed_path,group_path):
        self.text_bert.save_pretrained(text_embed_path)
        self.group_bert.save_pretrained(group_path)