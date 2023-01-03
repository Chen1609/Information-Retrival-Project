
from collections import Counter
import os
from pathlib import Path
import pickle


def PageViews(pv_path, docs):
    # read in the counter
    p = Path(pv_path) 
    pv_clean = f'{p.stem}.pkl'

    with open(pv_clean, 'rb') as f:
        wid2pv = pickle.loads(f.read())
        docs_views = {doc_id: wid2pv.get(doc_id, 0) for doc_id in docs}
    return docs_views
