# from transformers import TFBertForTokenClassification, BertForTokenClassification, BertTokenizer
from pytorch_transformers import BertForTokenClassification, BertTokenizer
# to remove
import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import English


# taken from the original repo
#  https://github.com/nickvosk/sigir2020-query-resolution

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.split(' ')
        tokens = []
        labels = []
        valid = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    valid.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # mask out labels for current turn.
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(valid) == max_seq_length
        return torch.tensor([input_ids], dtype=torch.long), torch.tensor([input_mask], dtype=torch.long), torch.tensor(
            [valid], dtype=torch.long), torch.tensor([segment_ids], dtype=torch.long)


class Ner(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)  # ,device='cuda')

        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class QuReTeCRewriter():
    def __init__(self, model_path,use_sep_token=True):
        self.model = Ner.from_pretrained(model_path).eval()
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        self.eng = English()
        self.sep_token=" [SEP] " if use_sep_token else " "

    def rewrite(self, query, **ctx):
        if len(ctx['history']) == 0:
            return query
        hist = " ".join(ctx["history"])
        query_input = query
        if 'canonical_rsp' in ctx and ctx['canonical_rsp'] is not None:
            merged_hist = []
            for i in range(len(ctx['history'])):
                merged_hist.append(hist[i])
                if ctx['canonical_rsp'][i] is not None:
                    merged_hist.append(ctx['canonical_rsp'][i])
            hist =" ".join(merged_hist)
        # hist=hist.replace(".",".?")
        # query_input=query_input.replace(".",".?")
        hist = " ".join([tok.text for tok in self.eng(hist)])
        query_tokens = [tok.text for tok in self.eng(query_input)]
        query_input = " ".join(query_tokens)
        src_text = "{} [SEP] {}".format(hist, query_input)
        input_id, mask, valid, tt_ids = convert_examples_to_features([src_text], 300, self.tokenizer)
        tokens = self.tokenizer.convert_ids_to_tokens(input_id.reshape(-1).numpy())

        logits = self.model(input_id, tt_ids, attention_mask=mask, valid_ids=valid)
        pred = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
        post_processed_tokens = []
        for t, v in zip(tokens, valid.reshape(-1).numpy().tolist()):
            if t == "[SEP]":
                break
            if v == 1:
                post_processed_tokens.append(t)
            else:
                post_processed_tokens[-1] += t.replace("##", '')
        tokens_to_add = [t for t, p in zip(post_processed_tokens, pred.reshape(-1).numpy()) if p == 2]
        print("token to add:", set(tokens_to_add))
        return query + self.sep_token + " ".join(set(tokens_to_add)) if len(tokens_to_add)>0 else query
