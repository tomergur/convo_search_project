from copy import deepcopy
from typing import List

from pygaggle.rerank.base import Reranker, Query, Text
from transformers import TFPreTrainedModel, PreTrainedTokenizer, TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import time
import datetime
class BertReranker(Reranker):
    def __init__(self,
                 model: TFPreTrainedModel = None,
                 tokenizer: PreTrainedTokenizer = None,
                 device: str =None,
                 strategy=None,
                 batch_size=32):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.batch_size = batch_size
        self.keras_model = self.create_keras_model()
        logs_dir = "./data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_dir,
                                                              histogram_freq=1,
                                                              profile_batch='2,4')
        self.device = device
        self.strategy=strategy

    def create_keras_model(self):
        input_ids = tf.keras.layers.Input(shape=(512,), name='input_token', dtype='int32')
        att_mask = tf.keras.layers.Input(shape=(512,), name='masked_token', dtype='int32')
        tt_ids = tf.keras.layers.Input(shape=(512,), name='token_type', dtype='int32')
        logits = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                            training=False)['logits']
        scores = tf.keras.layers.Activation(tf.nn.log_softmax)(logits)
        return tf.keras.Model(inputs=(input_ids, att_mask, tt_ids), outputs=scores)

    @staticmethod
    def get_model(pretrained_model_name_or_path: str = 'castorini/monobert-large-msmarco',
                  *args, device: str = None, **kwargs) -> TFAutoModelForSequenceClassification:
        # device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device(device)
        return TFAutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                    *args, **kwargs)

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str = 'bert-large-uncased',
                      *args, **kwargs) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs)

    def _rerank_keras(self, query: Query, texts: List[Text]) -> List[Text]:
        # print(tf.config.list_physical_devices('GPU'))
        data_prerpare_time = time.perf_counter()
        texts = deepcopy(texts)
        '''
        #ttexts=[tf.Tensor(t.text) for t in texts]
        ttexts = [t.text for t in texts]
        text_ds = tf.data.Dataset.from_tensor_slices(ttexts)
        q_ds = tf.data.Dataset.from_tensor_slices([query.text] * len(texts))
        pairs_ds = tf.data.Dataset.zip((q_ds, text_ds))
        pairs_ds = pairs_ds.map(lambda x,y: self.tokenizer(x.numpy(), y.numpy(), max_length=512,
                                                                   truncation=True, padding="max_length",
                                                                   return_token_type_ids=True, return_tensors='tf'))
        pairs_ds = pairs_ds.map(lambda x: (x['input_ids'], x['attention_mask'], x['token_type_ids']))
        '''
        ret = self.tokenizer([query.text] * len(texts),
                             [text.text for text in texts], max_length=512,
                             truncation=True, padding="max_length", return_token_type_ids=True, return_tensors='tf')
        input_ids = tf.data.Dataset.from_tensor_slices(ret['input_ids'])
        tt_ids = tf.data.Dataset.from_tensor_slices(ret['token_type_ids'])
        att_mask = tf.data.Dataset.from_tensor_slices(ret['attention_mask'])
        # pairs_ds=tf.data.Dataset.zip((input_ids,att_mask, tt_ids))
        pairs_ds=tf.data.Dataset.from_tensor_slices({"input_token": ret['input_ids'], "masked_token": ret['attention_mask'],
                                           "token_type": ret['token_type_ids']})
        rerank_time = time.perf_counter()
        print("data prepare time:", rerank_time - data_prerpare_time)
        '''
        scores = self.keras_model.predict({"input_token": ret['input_ids'], "masked_token": ret['attention_mask'],
                                           "token_type": ret['token_type_ids']}, batch_size=self.batch_size, verbose=1,
                                          callbacks=self.tboard_callback)
        '''
        with self.strategy.scope():
            scores = self.keras_model.predict(pairs_ds, verbose=1)
        # ,workers=2,use_multiprocessing=True
        print("batch infer time:", time.perf_counter() - rerank_time)
        # scores = scores.numpy()
        for i, text in enumerate(texts):
            text.score = scores[i, -1]
        print("total rerank time:{} sec".format(time.perf_counter() - data_prerpare_time))
        return texts

    def rerank2(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        infer_ttime = 0
        rerank_time = time.perf_counter()
        token_time = time.perf_counter()
        ret = self.tokenizer([query.text] * len(texts),
                             [text.text for text in texts], max_length=512,
                             truncation=True, padding=True, return_token_type_ids=True, return_tensors='tf')
        print("batch token time:", time.perf_counter()- token_time)
        input_ids_all = ret['input_ids']
        tt_ids_all = ret['token_type_ids']
        att_mask_all = ret['attention_mask']
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            input_ids = input_ids_all[i:i + self.batch_size]
            tt_ids = tt_ids_all[i:i + self.batch_size]
            att_mask = att_mask_all[i:i + self.batch_size]
            infer_time = time.perf_counter()
            output, = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                             training=False)
            scores = tf.nn.log_softmax(output)
            infer_ttime += time.perf_counter() - infer_time
            #print("batch infer time:", infer_ttime)
            scores = scores.numpy()
            for i, text in enumerate(batch_texts):
                text.score = scores[i, -1]
        print("batch infer time:", infer_ttime)
        print("total rerank time:{} sec".format(time.perf_counter() - rerank_time))
        return texts

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        if self.strategy:
            print("run strategy")
            return self._rerank_keras(query,texts)
        if self.device:
            with tf.device(self.device):
                return self._rerank(query,texts)
        return self._rerank(query,texts)

    def _rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        infer_ttime = 0
        token_ttime = 0
        rerank_time = time.perf_counter()
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            token_time = time.perf_counter()
            ret = self.tokenizer([query.text] * len(batch_texts),
                             [text.text for text in batch_texts], max_length=512,
                             truncation=True, padding=True, return_token_type_ids=True, return_tensors='tf')
            token_ttime += time.perf_counter() - token_time
            input_ids = ret['input_ids']
            tt_ids = ret['token_type_ids']
            att_mask = ret['attention_mask']
            infer_time = time.perf_counter()
            output, = self.model(input_ids, token_type_ids=tt_ids, attention_mask=att_mask, return_dict=False,
                             training=False)
            scores = tf.nn.log_softmax(output)
            infer_ttime += time.perf_counter() - infer_time
            #print("batch infer time:", infer_ttime)
            scores = scores.numpy()
            for i, text in enumerate(batch_texts):
                text.score = scores[i, -1]
        print("batch token time:", token_ttime)
        print("batch infer time:", infer_ttime)
        print("total rerank time:{} sec".format(time.perf_counter() - rerank_time))
        return texts


