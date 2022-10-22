import itertools

LAMBD_VALS=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,.9,1]
RBO_N_VALS=[5, 10, 25, 50, 100, 250, 500]
RBO_N_VALS=[5, 10, 25, 50, 100]
K_VALS=[5,10,25,50,100,250,500,1000]
K_VALS=[5,10,25,50,100]
DECAY_VALS=[0.01,1,5,10,100]
QPP_FEATURES_PARAMS={"WIG":[{"k":v} for v in K_VALS],
                     "clarity":[{"k":v} for v in K_VALS],
                     "NQC":[{"k":v} for v in K_VALS],
                     "NQC_norm":[{"k":v} for v in K_VALS],
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
                     "bert_qpp": [{"suffix": v} for v in ["_v1","_v2","_v3", "_v4"]],
                     "bert_qpp_hist": [{"suffix": v} for v in ["_v1","_v2","_v3", "_v4"]],
                     "bert_qpp_prev": [{"suffix": v} for v in ["_v1","_v2","_v3", "_v4"]],
                     "many_turns_bert_qpp": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                     "many_turns_bert_qpp_hist": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                     "many_turns_bert_qpp_prev": [{"suffix": v} for v in ["_v1", "_v2", "_v3", "_v4"]],
                     "many_turns_bert_qpp_cls": [{"suffix": v} for v in ["_v1", "_v2"]],
                     "many_turns_bert_qpp_reg": [{"suffix": v} for v in ["_v3", "_v4"]],
                     "many_turns_bert_qpp_hp": [{"suffix": v} for v in ["_v3", "_v4","_v11", "_v12"]],
                     "ref_hist_max_idf":[{"n":n,"lambd":lambd} for n,lambd in itertools.product(RBO_N_VALS,LAMBD_VALS)],
                     "ref_hist_avg_idf":[{"n":n,"lambd":lambd} for n,lambd in itertools.product(RBO_N_VALS,LAMBD_VALS)],
                     "ref_hist_max_scq":[{"n":n,"lambd":lambd} for n,lambd in itertools.product(RBO_N_VALS,LAMBD_VALS)],
                     "ref_hist_avg_scq":[{"n": n, "lambd": lambd} for n, lambd in itertools.product(RBO_N_VALS, LAMBD_VALS)],
                     "ref_hist_max_var":[{"n": n, "lambd": lambd} for n, lambd in itertools.product(RBO_N_VALS, LAMBD_VALS)],
                     "ref_hist_avg_var": [{"n": n, "lambd": lambd} for n, lambd in
                                          itertools.product(RBO_N_VALS, LAMBD_VALS)],
                     "ref_hist_bert_qpp": [{"n": n, "lambd": lambd} for n, lambd in
                                          itertools.product(RBO_N_VALS, LAMBD_VALS)],
                     "ref_hist_WIG": [{"k":k,"n": n, "lambd": lambd} for k,n, lambd in
                                          itertools.product(K_VALS,RBO_N_VALS, LAMBD_VALS)],
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
                     "ref_hist_decay_clarity_norm": [{"decay": d,"k":k,"lambd": lambd} for d,k, lambd in
                                            itertools.product(DECAY_VALS,K_VALS, LAMBD_VALS)],
                     "ref_hist_decay_NQC_norm": [{"decay": d,"k":k,"lambd": lambd} for d,k, lambd in
                                            itertools.product(DECAY_VALS,K_VALS, LAMBD_VALS)],
                     "ref_hist_decay_WIG_norm": [{"decay": d,"k":k,"lambd": lambd} for d,k, lambd in
                                            itertools.product(DECAY_VALS,K_VALS, LAMBD_VALS)],
                     "ref_hist_decay_norm_max_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                itertools.product(DECAY_VALS, LAMBD_VALS)],
                     "ref_hist_decay_norm_avg_idf": [{"decay": d, "lambd": lambd} for d, lambd in
                                                itertools.product(DECAY_VALS, LAMBD_VALS)],
                     "ref_hist_comb_decay_max_idf": [{"decay": d, "n": n, "lambd": lambd} for d,n, lambd in
                                                     itertools.product(DECAY_VALS,RBO_N_VALS, LAMBD_VALS)],
                     "ref_hist_comb_decay_avg_idf": [{"decay": d, "n": n, "lambd": lambd} for d,n, lambd in
                                                     itertools.product(DECAY_VALS,RBO_N_VALS, LAMBD_VALS)],
                     }

