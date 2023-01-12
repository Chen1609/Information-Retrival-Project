import search
from inverted_index import InvertedIndex
from Parser import process_wiki

import os
from pathlib import Path
import pickle


pkl_file = "part15_preprocessed.pkl"
try:
    if os.environ["assignment_2_data"] is not None:
      pkl_file = Path(os.environ["assignment_2_data"])
except:
   Exception("Problem with one of the variables")
   
assert os.path.exists(pkl_file), 'You must upload this file.'
with open(pkl_file, 'rb') as f:
  pages = pickle.load(f)  


all_words_body, all_words_title, all_words_anchor, docs = process_wiki(pages, 'all_words')

body_idx = InvertedIndex.read_index('body_indices', 'all_words')
title_idx = InvertedIndex.read_index('title_index', 'all_words')
anchor_idx = InvertedIndex.read_index('anchor_index', 'all_words')

import json
with open('queries_train.json') as f:
    data = json.load(f)

test_data = {q_id: d for q_id, d in enumerate(data.items())}
queries = {q_id: qf[0] for q_id, qf in test_data.items()}

import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
# nltk.download('stopwords')


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
stopwords_frozen = frozenset(stopwords.words('english'))
def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.
    
    Parameters:
    -----------
    text: string , represting the text to tokenize.    
    
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens =  [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in stopwords_frozen]    
    return list_of_tokens



from search import Search
from cosineSimilarity import Cosine_Similarity
from binarySimilarity import Binary_Similarity
from LSISimilairty import LSI_Similarity

lsi = LSI_Similarity(anchor_idx, 10)

queries_tokenize = {q_id: tokenize(q) for q_id, q in queries.items()}
search = Search()
# tfidf_queries_score_train = search.get_topN_score_for_queries(queries_tokenize, title_idx,N=30, score=Binary_Similarity)
# tfidf_queries_score = search.get_topN_score_for_queries(queries_tokenize, anchor_idx,N=30, score=LSI_Similarity)
print(len(lsi.get_page_views_filterd_threashold(docs)))