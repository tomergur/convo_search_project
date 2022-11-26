import tensorflow as tf
import keras_nlp


class BertPL(tf.keras.Model):
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
        text_emb = tf.reshape(bert_res.pooler_output, [num_seq, seq_length, self.text_encoder.config.hidden_size])
        #tf.print("text emb shape",tf.shape(text_emb))
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
        self.chunk_encoder.save_weights('{}/chunk_encoder.ckp'.format(model_path))




