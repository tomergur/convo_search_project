
class DirSmoothLM:
    def __init__(self, mu, doc_tf, doc_len, col_lm):
        self.mu = mu
        self.doc_tf = doc_tf
        self.doc_len = doc_len
        self.col_lm = col_lm

    def __getitem__(self, term):
        if self.col_lm[term] == 0.:
            return 0
        return (self.mu * self.col_lm[term] + self.doc_tf.get(term, 0)) / (self.doc_len + self.mu)

    def get_all_terms(self):
        return self.doc_tf.keys()

    def get_doc_weight(self, term):
        unseen_coef = 1 - float(self.mu) / float(self.mu + self.doc_len)
        return unseen_coef * self.doc_tf.get(term, 0) / self.doc_len