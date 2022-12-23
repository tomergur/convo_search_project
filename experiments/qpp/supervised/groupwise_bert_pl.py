import tensorflow as tf
import keras_nlp
from transformers import TFAutoModel,BertConfig,TFBertForTokenClassification
import os
class GroupwiseBertPL(tf.keras.Model):

    @staticmethod
    def from_pretrained(model_path,lists_lengths):
        text_embed_path = "{}/text_embed".format(model_path)
        group_path = "{}/group_model".format(model_path)
        lists_encoder_path="{}/lists_encoder".format(model_path)
        text_model = TFAutoModel.from_pretrained(text_embed_path)
        lstm = tf.keras.Sequential([tf.keras.layers.LSTM(text_model.config.hidden_size, dropout=0.2, time_major=False)])
        lstm.build(input_shape=(1, None, text_model.config.hidden_size))
        lstm.load_weights('{}/chunk_encoder.h5'.format(lists_encoder_path))

    @staticmethod
    def create_model(text_embed_model_name,groupwise_hidden_layers,lists_lengths):
        text_model=TFAutoModel.from_pretrained(text_embed_model_name)
        lstm = tf.keras.Sequential([tf.keras.layers.LSTM(text_model.config.hidden_size, dropout=0.2, time_major=False)])
        group_conf = BertConfig(num_hidden_layers=groupwise_hidden_layers, num_labels=2)
        groupwise_model =  TFBertForTokenClassification(group_conf)
        return GroupwiseBertPL(text_model,lstm,groupwise_model,lists_lengths)

    def __init__(self,text_encoder,list_encoder,groupwise_model,lists_lengths):
        super(GroupwiseBertPL, self).__init__()
        self.text_encoder=text_encoder
        self.groupwise_model=groupwise_model
        self.list_encoder=list_encoder
        self.lists_lengths=lists_lengths
        self.pos_encoding = keras_nlp.layers.SinePositionEncoding()

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs['input_ids'])
        num_seq = input_shape[0]
        seq_length = input_shape[1]
        entry_length = input_shape[2]
        inputs = {k: tf.reshape(v, [-1, entry_length]) for k, v in inputs.items()}
        bert_res = self.text_encoder(**inputs, training=training)
        num_lists=seq_length//self.lists_lengths
        #tf.print("num lists",num_lists)
        text_emb = tf.reshape(bert_res.pooler_output, [num_seq*num_lists,self.lists_lengths, self.text_encoder.config.hidden_size])
        #tf.print(tf.shape(text_emb))
        text_emb=text_emb+self.pos_encoding(text_emb)
        lists_embedding=self.list_encoder(text_emb,training=training)
        #tf.print(tf.shape(lists_embedding))
        lists_embedding=tf.reshape(lists_embedding,[num_seq,num_lists,self.text_encoder.config.hidden_size])
        group_inputs = {'inputs_embeds': lists_embedding}
        res = self.groupwise_model(group_inputs, training=training)
        return res.logits

    def save_pretrained(self,output_path):
        text_embed_path = "{}/text_embed".format(output_path)
        group_path = "{}/group_model".format(output_path)
        lists_encoder_path="{}/lists_encoder".format(output_path)
        self.text_encoder.save_pretrained(text_embed_path)
        self.groupwise_model.save_pretrained(group_path)
        if not os.path.exists(lists_encoder_path):
            os.mkdir(lists_encoder_path)
        self.list_encoder.save_weights('{}/chunk_encoder.h5'.format(lists_encoder_path))



