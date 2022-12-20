import tensorflow as tf
import keras_nlp
from transformers import TFAutoModel

class BertPL(tf.keras.Model):
    @staticmethod
    def from_pretrained(model_path, chunk_size):
        # group_model = TFAutoModelForTokenClassification.from_pretrained(group_model_path)
        # "groupwise_bert"
        text_model_path = model_path + "/text_embed/"
        text_model = TFAutoModel.from_pretrained(text_model_path)
        bert_pl=BertPL(text_model,chunk_size)
        bert_pl.chunk_encoder.build(input_shape=(1, None, text_model.config.hidden_size))
        bert_pl.chunk_encoder.load_weights('{}/chunk_encoder.h5'.format(model_path))
        return bert_pl

    def __init__(self,text_encoder,chunk_size):
        super(BertPL, self).__init__()
        self.text_encoder=text_encoder
        self.pos_encoding=keras_nlp.layers.SinePositionEncoding()
        lstm=tf.keras.layers.LSTM(self.text_encoder.config.hidden_size,dropout=0.2,time_major=False)
        dense=tf.keras.layers.Dense(100, activation='relu')
        classfication_layer=tf.keras.layers.Dense(chunk_size+1, activation='relu')
        self.chunk_encoder=tf.keras.Sequential([lstm,dense,classfication_layer])
        self.chunk_size=chunk_size

    def call(self, inputs, training=False):
        input_shape = tf.shape(inputs['input_ids'])
        num_seq = input_shape[0]
        seq_length = input_shape[1]
        entry_length = input_shape[2]
        #tf.print("input shape", tf.shape(inputs['input_ids']))
        inputs = {k: tf.reshape(v, [-1, entry_length]) for k, v in inputs.items()}
        bert_res = self.text_encoder(**inputs, training=training)
        ####chunking
        #bert_res=tf.map_fn(lambda i:self.text_encoder(**{k:v[i:i+self.chunk_size,:] for k, v in inputs.items()}, training=training).pooler_output,tf.range(num_seq*seq_length,delta=self.chunk_size),fn_output_signature=tf.float32,parallel_iterations=1)
        #text_emb = tf.reshape(bert_res, [num_seq, seq_length, self.text_encoder.config.hidden_size])
        ### chunking res
        text_emb = tf.reshape(bert_res.pooler_output, [num_seq, seq_length, self.text_encoder.config.hidden_size])
        text_emb=text_emb+self.pos_encoding(text_emb)
        num_chunks = (seq_length * num_seq) / self.chunk_size
        text_emb = tf.reshape(text_emb, [num_chunks, self.chunk_size, self.text_encoder.config.hidden_size])
        #tf.print(tf.shape(text_emb))
        #lstm_output = self.lstm(text_emb)
        return self.chunk_encoder(text_emb)

    def save_pretrained(self,model_path):
        text_enc_path=model_path+"/text_embed/"
        self.text_encoder.save_pretrained(text_enc_path)
        #self.lstm.save_weights('{}/lstm.ckp'.format(model_path))
        self.chunk_encoder.save_weights('{}/chunk_encoder.h5'.format(model_path))




