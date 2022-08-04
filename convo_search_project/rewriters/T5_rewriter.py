from transformers import T5Tokenizer, TFT5ForConditionalGeneration

from spacy.lang.en import English


class T5Rewriter():
    def __init__(self, model_str, num_queries_generated=1, context_window=None, selected_query_rank=1, from_pt=True,
                 append_history=False, max_length=64, num_beams=10, sliding_window_fusion=False,
                 early_stopping=True, rsp_context_type="type_b"):
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_str, from_pt=from_pt)
        self.tokenizer = T5Tokenizer.from_pretrained(model_str)
        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.context_window = context_window
        self.selected_query_rank = selected_query_rank
        self.sliding_window_fusion = sliding_window_fusion
        self.num_queries_generated = num_queries_generated
        self.eng = English()
        self.append_history = append_history
        self.rsp_context_type = rsp_context_type
        # self.ntr = Ntr()

    def rewrite(self, query, **ctx):
        if len(ctx['history']) == 0:
            return query
        if self.sliding_window_fusion:
            return self._sliding_history_rewrite(ctx['history'], query)
        if self.context_window:
            history = ctx['history'][-self.context_window:]
            print("history:", history)
        else:
            history = ctx['history']
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            if self.rsp_context_type == "type_b":
                history = history + [ctx["canonical_rsp"][-1]]
            else:
                merged_hist = []
                for i in range(len(ctx['history'])):
                    merged_hist.append(history[i])
                    merged_hist.append(ctx['canonical_rsp'][i])
                history = merged_hist
        return self._generate_queries(history, query, self.num_queries_generated)

    def _sliding_history_rewrite(self, history, query):
        res = []
        for i in range(min(self.num_queries_generated, len(history))):
            res.append(self._generate_queries(history[i:], query, 1))
        return res

    def _generate_queries(self, history, query, num_queries_generated):
        src_text = " ||| ".join(history + [query])
        src_text = " ".join([tok.text for tok in self.eng(src_text)])
        input_ids = self.tokenizer(
            src_text, return_tensors="tf", add_special_tokens=True)
        selected_query_offset = self.selected_query_rank - 1
        '''
        if self.use_sampling:
            print("use sampling",num_queries_generated + selected_query_offset)
            output_ids = self.model.generate(input_ids['input_ids'], max_length=self.max_length,
                                             num_return_sequences=num_queries_generated + selected_query_offset,
                                             do_sample=True, top_p=0.95,temperature=3)
       '''
        output_ids = self.model.generate(
            input_ids['input_ids'],
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
            num_return_sequences=num_queries_generated + selected_query_offset)
        # Decode output
        query_rewrites = []
        for i in range(selected_query_offset, num_queries_generated + selected_query_offset):
            query_rewrite = self.tokenizer.decode(
                output_ids[i, 0:],
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True)
            if self.append_history:
                # history_sep=" "
                history_sep = " [SEP] "
                history_query = history_sep.join(history + [query])
                query_rewrite = " [SEP] ".join([history_query, query_rewrite])
            query_rewrites.append(query_rewrite)
        return query_rewrites[0] if len(query_rewrites) == 1 else query_rewrites
