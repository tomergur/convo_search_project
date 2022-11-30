import tensorflow as tf
import keras
import keras_nlp
from transformers import TFAutoModel

class SeqQPP(tf.keras.Model):
    @staticmethod
    def from_pretrained(model_path):
        # group_model = TFAutoModelForTokenClassification.from_pretrained(group_model_path)
        # "groupwise_bert"
        text_model_path = model_path + "/text_embed/"
        text_model = TFAutoModel.from_pretrained(text_model_path)
        #chunk_encoder=keras.models.load_model('{}/chunk_encoder.h5'.format(model_path))
        bert_pl=SeqQPP(text_model,None)
        bert_pl.chunk_encoder.build(input_shape= (1,None,text_model.config.hidden_size))
        bert_pl.chunk_encoder.load_weights('{}/chunk_encoder.h5'.format(model_path))
        return bert_pl

    def __init__(self,text_encoder,chunk_encoder=None):
        super(SeqQPP, self).__init__()
        self.text_encoder=text_encoder
        self.pos_encoding=keras_nlp.layers.SinePositionEncoding()
        if chunk_encoder is None:
            lstm=tf.keras.layers.LSTM(self.text_encoder.config.hidden_size,dropout=0.2,time_major=False,return_sequences=True)
            dense=tf.keras.layers.Dense(100, activation='relu')
            classfication_layer=tf.keras.layers.Dense(2, activation='relu')
            self.chunk_encoder=tf.keras.Sequential([lstm,dense,classfication_layer])
        else:
            self.chunk_encoder=chunk_encoder

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs['input_ids'])
        num_seq = input_shape[0]
        seq_length = input_shape[1]
        entry_length = input_shape[2]
        inputs = {k: tf.reshape(v, [-1, entry_length]) for k, v in inputs.items()}
        bert_res = self.text_encoder(**inputs, training=training)
        text_emb = tf.reshape(bert_res.pooler_output, [num_seq, seq_length, self.text_encoder.config.hidden_size])
        text_emb=text_emb+self.pos_encoding(text_emb)
        return self.chunk_encoder(text_emb)

    def save_pretrained(self,model_path):
        text_enc_path=model_path+"/text_embed/"
        self.text_encoder.save_pretrained(text_enc_path)
        self.chunk_encoder.save_weights('{}/chunk_encoder.h5'.format(model_path))




