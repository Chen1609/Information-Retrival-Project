BLOCK_SIZE = 3999998 # == 2MB-2bytes, a number that is divisible by TUPLE_SIZE.

from collections import Counter, defaultdict
from itertools import groupby, chain
import os
from pathlib import Path
import pickle
import re
import pandas as pd
from inverted_index import InvertedIndex

import json
import nltk
from nltk.corpus import stopwords

from pageViews import PageViews

nltk.download('stopwords')

# Each page is a tuple in the form of (page_id, title, body, [(target_page_id, anchor_text), ...])


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

pkl_file = "part15_preprocessed.pkl"
try:
    if os.environ["assignment_2_data"] is not None:
      pkl_file = Path(os.environ["assignment_2_data"])
except:
   Exception("Problem with one of the variables")
   
assert os.path.exists(pkl_file), 'You must upload this file.'
with open(pkl_file, 'rb') as f:
  pages = pickle.load(f)  


RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)
def tokenize(text):
  return [token.group() for token in RE_WORD.finditer(text.lower())]


def count_words(pages):
  """ Count words in the text of articles' title, body, and anchor text using 
      the above `tokenize` function. 
  Parameters:
  -----------
  pages: list of tuples
    Each tuple is a wiki article with id, title, body, and 
    [(target_article_id, anchor_text), ...]. 
  Returns:
  --------
  list of str
    A list of tokens
  """
  word_counts = Counter()
  for wiki_id, title, body, links in pages:
    tokens_title = tokenize(title)
    tokens_body = tokenize(body)
    # tokens_wiki_id = tokenize(wiki_id)
    # tokenize body and anchor text and count
    # YOUR CODE HERE
    title_counter = Counter(tokens_title)
    body_counter = Counter(tokens_body)
    # wiki_id_counter = Counter(tokens_wiki_id)
    word_counts += title_counter + body_counter
  return word_counts

  # create directories for the different indices 

# default tokenizer keeping all words
default_tokenizer = lambda text: tokenize(text)

try:
  os.mkdir('body_indices')
except:
  pass
try:
  os.mkdir('title_index')
except:
  pass
try:
  os.mkdir('anchor_index')
except:
  pass
# define batch iterator
def batch_iterator(it, batch_size=1000):
  """ Generator that yields items in a batch. Yields the batch 
      index (0, 1, ..) and an iterable of items in each batch of `batch_size` 
      in length. 
  """
  for i, group in groupby(enumerate(it), 
                          lambda x: x[0] // batch_size):
    _, batch = zip(*group)
    yield i, batch


def process_wiki(pages, index_name, tokenize_func=default_tokenizer):
  """ Process wikipedia: tokenize article body, title, anchor text, and create
      indices for them. Each index is named `index_name` and placed in a 
      directory under the current dir named 'body_indices', 'title_index' 
      and 'anchor_index', respectively. 
  Parameters:
  -----------
  pages: list of tuples
    Each tuple is a wiki article with id, title, body, and 
    [(target_article_id, anchor_text), ...]. 
  index_name: str
    The name for the index.
  tokenize_func: function str -> list of str
    Tokenization function that takes text as input and return a list of 
    tokens.
  Returns:
  --------
  Three inverted index objects
    body_index, title_index, anchor_index.
  """
  docs = set()
  # create the index for titles
  title_index = InvertedIndex()
  # collect anchor text tokens for each target article by its id
  id2anchor_text = defaultdict(list)

  # iterate over batches of pages from the dump
  body_index_names = []
  counter = 0
  for batch_idx, batch_pages in batch_iterator(pages):
    ids, titles, bodies, links = zip(*batch_pages)
    target_ids, anchor_texts = zip(*[wl for l in links for wl in l])
    # tokenize
    titles = map(tokenize_func, titles)
    bodies = map(tokenize_func, bodies)
    anchor_texts = map(tokenize_func, anchor_texts)
    # create a separate index of articles body for article in this batch
    body_index = InvertedIndex()
    for id, title, body in zip(ids, titles, bodies):
      title_index.add_doc(id, title)
      body_index.add_doc(id, body)
      docs.add(id)
      
    for target_id, anchor_text in zip(target_ids, anchor_texts):
      id2anchor_text[target_id].extend(anchor_text)
    body_index.write('./body_indices', f'{index_name}_{batch_idx}')
    body_index_names.append(f'{index_name}_{batch_idx}')
  # merge body indices from the different batches into one index and delete 
  # the parts
  body_index = InvertedIndex()
  body_index.merge_indices('./body_indices', 
                           body_index_names, index_name)
  for idx_name in body_index_names:
    InvertedIndex.delete_index('./body_indices', idx_name)
  title_index.write('./title_index', index_name)
  # create index for anchor text
  anchor_index = InvertedIndex()
  for id, tokens in id2anchor_text.items():
    anchor_index.add_doc(id, tokens)
  anchor_index.write('./anchor_index', index_name)
  return body_index, title_index, anchor_index, docs

