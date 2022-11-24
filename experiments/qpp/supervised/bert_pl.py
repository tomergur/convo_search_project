import tensorflow as tf
import keras_nlp
class BertPL(tf.keras.Model):
    def __init__(self,text_encoder,chunk_size):
        super().__init__()
        self.text_encoder=text_encoder
        self.pos_encoding=keras_nlp.layers.SinePositionEncoding()
        self.lstm=tf.keras.layers.LSTM(self.bert.config.hidden_size,dropout=0.2,time_major=False)
        self.dense=tf.keras.layers.Dense(100, activation='relu')
        self.output=tf.keras.layers.Dense(chunk_size+1, activation='relu')
        self.chunk_size=chunk_size

    def forward(self, inputs, training=False):
        input_shape = tf.shape(inputs['input_ids'])
        num_seq = input_shape[0]
        seq_length = input_shape[1]
        entry_length = input_shape[2]
        inputs = {k: tf.reshape(v, [-1, entry_length]) for k, v in inputs.items()}
        # tf.print(tf.shape(inputs['input_ids']))
        bert_res = self.text_encoder(**inputs, training=training)
        num_chunks = (seq_length*num_seq)/self.chunk_size
        text_emb = tf.reshape(bert_res.pooler_output, [num_chunks, self.chunk_size, -1])





