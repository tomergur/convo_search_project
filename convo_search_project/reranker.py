from copy import deepcopy
from typing import List

from pygaggle.rerank.base import Reranker, Query, Text
from transformers import TFPreTrainedModel, PreTrainedTokenizer, TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import time

class BertReranker(Reranker):
    def __init__(self,
                 model: TFPreTrainedModel = None,
                 tokenizer: PreTrainedTokenizer = None,
                 batch_size=32):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.batch_size = batch_size
        # self.device = next(self.model.parameters(), None).device

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

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        # print(tf.config.list_physical_devices('GPU'))
        texts = deepcopy(texts)
        device_str = '/GPU:0'
        device_str = '/GPU:1'
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            '''
            ret = self.tokenizer.encode_plus(query.text,
                                                   batch_texts[0].text,
                                                   max_length=512,
                                                   truncation=True,
                                                   return_token_type_ids=True,
                                                   return_tensors='tf')

            '''
            token_time = time.time()
            ret = self.tokenizer([query.text] * len(batch_texts),
                                 [text.text for text in batch_texts], max_length=512,
                                 truncation=True,padding=True, return_tensors='tf')
            #print("batch token time:",time.time()-token_time)

            input_ids = ret['input_ids']
            tt_ids = ret['token_type_ids']
            att_mask = ret['attention_mask']
            #infer_time=time.time()
            output, = self.model(input_ids, token_type_ids=tt_ids,attention_mask=att_mask, return_dict=False,training=False)
            scores = tf.nn.log_softmax(output)
            #print("batch infer time:", time.time() - infer_time)
            for i, text in enumerate(batch_texts):
                text.score = scores[i, -1].numpy()
        return texts
