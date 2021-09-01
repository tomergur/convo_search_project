from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# to remove
from chatty_goose.cqr.ntr import Ntr
from spacy.lang.en import English


class T5Rewriter():
    def __init__(self, model_str, num_queries_generated=1, from_pt=True, max_length=64, num_beams=10,
                 early_stopping=True):
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_str, from_pt=from_pt)
        self.tokenizer = T5Tokenizer.from_pretrained(model_str)
        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.num_queries_generated = num_queries_generated
        self.eng = English()
        # self.ntr = Ntr()

    def rewrite(self, query, **ctx):
        if len(ctx['history']) == 0:
            return query
        src_text = " ||| ".join(ctx['history'] + [query])
        # prev_src = src_text
        src_text = " ".join([tok.text for tok in self.eng(src_text)])
        input_ids = self.tokenizer(
            src_text, return_tensors="tf", add_special_tokens=True
        )
        output_ids = self.model.generate(
            input_ids['input_ids'],
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
            num_return_sequences=self.num_queries_generated
        )

        # Decode output
        query_rewrites = []
        for i in range(self.num_queries_generated):
            query_rewrite = self.tokenizer.decode(
                output_ids[i, 0:],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True)
            query_rewrites.append(query_rewrite)
        # remove later
        # ntr_query = self.Ntr.rewrite(query)
        # print(ntr_query, query_rewrites[0])
        # assert ntr_query == query_rewrites[0]
        return query_rewrites[0] if len(query_rewrites) == 1 else query_rewrites
