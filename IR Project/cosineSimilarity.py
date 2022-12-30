import numpy as np
def Cosine_Similarity(d,q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores 
    key: doc_id
    value: cosine similarity score
    
    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores
    
    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                           value: cosine similarty score.
    """
    # YOUR CODE HERE
    q_norm = np.linalg.norm(q)
    d_norm = np.sqrt(np.square(d).sum(axis=1))
    dot_product = d.dot(q)
    cos_sim = dot_product / (q_norm * d_norm)
    cos_sim_dict =  cos_sim.to_dict()
    return cos_sim_dict