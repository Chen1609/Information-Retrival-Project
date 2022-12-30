import pandas as pd

def Binary_Similarity(d, q):
    d_bool = d.astype(bool)
    q_bool = q.astype(bool)
    
    d_and_q = d_bool & q_bool
    scores = d_and_q.sum(axis=1)

    return scores.to_dict()

    
