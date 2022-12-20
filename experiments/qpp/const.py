import itertools

LAMBD_VALS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, .9, 1]
RBO_N_VALS = [5, 10, 25, 50, 100, 250, 500]
RBO_N_VALS = [5, 10, 25, 50, 100]
K_VALS = [5, 10, 25, 50, 100, 250, 500, 1000]
K_VALS = [5, 10, 25, 50, 100]
DECAY_VALS = [0.01, 1, 5, 10, 100]
QPP_FEATURES_PARAMS = {"WIG": [{"k": v} for v in K_VALS],
                       "clarity": [{"k": v} for v in K_VALS],
                       "NQC": [{"k": v} for v in K_VALS],
                       "NQC_norm": [{"k": v} for v in K_VALS],
                       "WIG_norm": [{"k": v} for v in K_VALS],
                       "clarity_norm": [{"k": v} for v in K_VALS],
                       "ref_rewrites_max_idf": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_avg_idf": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_max_scq": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_avg_scq": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_max_var": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_avg_var": [{"n": n, "lambd": lambd} for n, lambd in
                                                itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_WIG": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                            itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_clarity": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                                itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_NQC": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                            itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_WIG_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                                 itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_clarity_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                                     itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_rewrites_NQC_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                                 itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "bert_qpp_topiocqa": [{"suffix": v} for v in
                                    ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2", "_v2_3", "_v2_4",
                                     "_v2_5"]],
                       "bert_qpp_or_quac": [{"suffix": v} for v in
                                    ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2", "_v2_3", "_v2_4",
                                     "_v2_5"]],

                       "bert_qpp": [{"suffix": v} for v in ["_v1_1","_v1_2","_v1_3","_v1_4","_v1_5", "_v2_1","_v2_2","_v2_3","_v2_4","_v2_5"]],
                       "bert_qpp_3": [{"suffix": v} for v in
                                    ["_v1_1", "_v1_2", "_v1_3", "_v2_1", "_v2_2", "_v2_3"]],

                       "bert_pl": [{"suffix": v} for v in ["_v1_1","_v2_1"]],
                       "seq_qpp": [{"suffix": v} for v in [ "_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5","_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"]],
                       "seq_qpp_3": [{"suffix": v} for v in
                                   ["_v1_1", "_v1_2", "_v1_3", "_v2_1", "_v2_2", "_v2_3"]],
                       "rewrites_bert_qpp": [{"suffix": v} for v in
                                   ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"]],

                       "bert_qpp_cls": [{"suffix": v} for v in ["_v1", "_v2"]],
                       "bert_qpp_hist": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                       "st_bert_qpp": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                       "bert_qpp_prev": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                       "many_turns_bert_qpp": [{"suffix": v} for v in ["_v1", "_v2"]],
                       "many_turns_bert_qpp_cls": [{"suffix": v} for v in
                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2",
                                                       "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_or_quac": [{"suffix": v} for v in
                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2",
                                                       "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_topiocqa": [{"suffix": v} for v in
                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2",
                                                       "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens": [{"suffix": v} for v in
                                                        ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_l2": [{"suffix": v} for v in
                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2",
                                                       "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_1kturns": [{"suffix": v} for v in
                                                              ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                               "_v2_2",
                                                               "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_2kturns": [{"suffix": v} for v in
                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1", "_v2_2",
                                                       "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_3kturns": [{"suffix": v} for v in
                                                              ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                               "_v2_2",
                                                               "_v2_3", "_v2_4", "_v2_5"]],
                       "many_turns_bert_qpp_tokens_skturns": [{"suffix": v,"max_seq_length":k} for v,k in
                                                              itertools.product(["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                               "_v2_2",
                                                               "_v2_3", "_v2_4", "_v2_5"],[1,2,3,None])],
                       "many_turns_bert_qpp_tokens_3": [{"suffix": v} for v in ["_v1_1","_v1_2","_v1_3","_v2_1","_v2_2","_v2_3"]],
                       "many_turns_bert_qpp_tokens_init": [{"suffix": v} for v in
                                                      ["_v4_1", "_v4_2", "_v4_3", "_v4_4", "_v4_5"]],

                       "many_turns_bert_qpp_hist": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                       "many_turns_bert_qpp_prev": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                       "many_turns_bert_qpp_reg": [{"suffix": v} for v in ["_v3", "_v4"]],
                       "many_docs_bert_qpp": [{"suffix": "v1", "group_agg_func": "max"},
                                              {"suffix": "v2", "group_agg_func": "max"},
                                              {"suffix": "v3", "group_agg_func": "mean"},
                                              {"suffix": "v4", "group_agg_func": "mean"},
                                              {"suffix":"v5","group_agg_func":"first"},
                                              {"suffix":"v6","group_agg_func":"first"}],
                       "ref_hist_max_idf": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_avg_idf": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_max_scq": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_avg_scq": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_max_var": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_avg_var": [{"n": n, "lambd": lambd} for n, lambd in
                                            itertools.product(RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_bert_qpp": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS, ["_v1_1","_v1_2","_v1_3","_v1_4","_v1_5", "_v2_1","_v2_2","_v2_3","_v2_4","_v2_5"])],
                       "ref_hist_bert_qpp_or_quac": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                               ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                                "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_bert_qpp_topiocqa": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                               ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                                "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_bert_qpp_1kturns": [{"n": n, "lambd": lambd, "suffix": v,"ref_limit":1} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                               ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5", "_v2_1",
                                                                "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_bert_qpp_skturns": [{"n": n, "lambd": lambd, "suffix": v, "ref_limit": k} for
                                                     n, lambd, v,k in
                                                     itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                       ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                        "_v2_1",
                                                                        "_v2_2", "_v2_3", "_v2_4", "_v2_5"],[1,2,3,None])],
                       "ref_hist_bert_qpp_2kturns": [{"n": n, "lambd": lambd, "suffix": v, "ref_limit": 2} for
                                                    n, lambd, v in
                                                    itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                      ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                       "_v2_1",
                                                                       "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_bert_qpp_3kturns": [{"n": n, "lambd": lambd, "suffix": v, "ref_limit": 3} for
                                                     n, lambd, v in
                                                     itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                       ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                        "_v2_1",
                                                                        "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_rewrites_bert_qpp": [{"n": n, "lambd": lambd, "suffix": v, "ref_limit": 3} for
                                                     n, lambd, v in
                                                     itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                       ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                        "_v2_1",
                                                                        "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_bert_qpp_cls": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                               ["_v1", "_v2",])],
                       "ref_rewrites_bert_qpp": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                             itertools.product(RBO_N_VALS, LAMBD_VALS, ["_v1_1","_v1_2","_v1_3","_v1_4","_v1_5", "_v2_1","_v2_2","_v2_3","_v2_4","_v2_5"])],
                       "ref_rewrites_bert_qpp_all_methods": [{"n": n, "lambd": lambd, "suffix": v,"method_type":["all"]} for n, lambd, v in
                                                 itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                   ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                    "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_rewrites_bert_qpp_t5_methods": [{"n": n, "lambd": lambd, "suffix": v,"method_type":["t5"]} for n, lambd, v in
                                                     itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                       ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                        "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_rewrites_bert_qpp_hqe_methods": [{"n": n, "lambd": lambd, "suffix": v,"method_type":["hqe"]} for n, lambd, v in
                                                     itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                       ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                        "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_rewrites_bert_qpp_quretec_methods": [{"n": n, "lambd": lambd, "suffix": v,"method_type":["quretec"]} for n, lambd, v in
                                                             itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                               ["_v1_1", "_v1_2", "_v1_3", "_v1_4",
                                                                                "_v1_5",
                                                                                "_v2_1", "_v2_2", "_v2_3", "_v2_4",
                                                                                "_v2_5"])],


                       "ref_rewrites_many_turns_bert_qpp_tokens": [{"n": n, "lambd": lambd, "suffix": v} for n, lambd, v in
                                                 itertools.product(RBO_N_VALS, LAMBD_VALS,
                                                                   ["_v1_1", "_v1_2", "_v1_3", "_v1_4", "_v1_5",
                                                                    "_v2_1", "_v2_2", "_v2_3", "_v2_4", "_v2_5"])],
                       "ref_hist_WIG": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                        itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_clarity": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                            itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_NQC": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                        itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_WIG_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                             itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_clarity_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                                 itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_NQC_norm": [{"k": k, "n": n, "lambd": lambd} for k, n, lambd in
                                             itertools.product(K_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_decay_max_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_avg_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_max_scq": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_avg_scq": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_max_var": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_avg_var": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_clarity": [{"decay": d, "lambd": lambd} for d, lambd in
                                                  itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_NQC": [{"decay": d, "lambd": lambd} for d, lambd in
                                              itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_WIG": [{"decay": d, "lambd": lambd} for d, lambd in
                                              itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_clarity_norm": [{"decay": d, "k": k, "lambd": lambd} for d, k, lambd in
                                                       itertools.product(DECAY_VALS, K_VALS, LAMBD_VALS)],
                       "ref_hist_decay_NQC_norm": [{"decay": d, "k": k, "lambd": lambd} for d, k, lambd in
                                                   itertools.product(DECAY_VALS, K_VALS, LAMBD_VALS)],
                       "ref_hist_decay_WIG_norm": [{"decay": d, "k": k, "lambd": lambd} for d, k, lambd in
                                                   itertools.product(DECAY_VALS, K_VALS, LAMBD_VALS)],
                       "ref_hist_decay_norm_max_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                       itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_decay_norm_avg_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                       itertools.product(DECAY_VALS, LAMBD_VALS)],
                       "ref_hist_comb_decay_max_idf": [{"decay": d, "n": n, "lambd": lambd} for d, n, lambd in
                                                       itertools.product(DECAY_VALS, RBO_N_VALS, LAMBD_VALS)],
                       "ref_hist_comb_decay_avg_idf": [{"decay": d, "n": n, "lambd": lambd} for d, n, lambd in
                                                       itertools.product(DECAY_VALS, RBO_N_VALS, LAMBD_VALS)],
                       }
