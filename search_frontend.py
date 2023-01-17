
import math

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import bz2
from functools import partial
from collections import Counter, OrderedDict
import pickle
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
import os
import re
import gzip
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import itertools
from time import time
import hashlib
from inverted_index_gcp import *
import csv

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        self.inverted_index_title = InvertedIndex.read_index('title', 'title_index')
        self.inverted_index_body = InvertedIndex.read_index('body', 'body_index')
        self.inverted_index_anchor = InvertedIndex.read_index('anchor', 'anchor_index')

        self.bm25_body = BM25_from_index(self.inverted_index_body, base_dir='body')
        self.bm25_anchor = BM25_from_index(self.inverted_index_anchor, base_dir='anchor')
        self.bm25_title = BM25_from_index(self.inverted_index_title, base_dir='title')
        
        with open(Path('.') / f'term_total_title.pkl', 'rb') as f:
          self.inverted_index_title.term_total = pickle.load(f)
        with open(Path('.') / f'term_total_anchor.pkl', 'rb') as f:
          self.inverted_index_anchor.term_total = pickle.load(f)
        with open(Path('.') / f'term_total_index.pkl', 'rb') as f:
          self.inverted_index_body.term_total = pickle.load(f)
        
        with open('pageRank.csv') as pr:
          self.pageRank = dict(filter(None, csv.reader(pr)))
        with open('pageviews-202108-user.pkl', 'rb') as f:
          self.pageViews = pickle.loads(f.read())
        with open(Path('.') / f'wikiId_titles.pkl', 'rb') as f:
          self.wikiid_titles = pickle.load(f)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)



nltk.download('stopwords', quiet=True)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    t_query = {1 :tokenize(query)}
    # res_body = app.bm25_body.search(t_query, N=20)
    # res_anchor = app.bm25_anchor.search(t_query, N=50)
    res_title = app.bm25_title.search(t_query, N=70)
    # bm25_t055_b015_a030 = merge_results(title_scores=res_title, N=70)
    
    res = get_page_title(res_title[1].keys())
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokenize = {1: tokenize(query)}
    res = get_page_title(\
      map(lambda results: results[0], \
        get_topN_score_for_queries(query_tokenize, app.inverted_index_body, N=100, score=cosine_similarity, base_dir='body')[1]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokenize = {1: tokenize(query)}
    res = get_page_title(\
      map(lambda results: results[0], \
         get_topN_score_for_queries(query_tokenize, app.inverted_index_title, N=100, score=binary_similarity, base_dir='title')[1]))
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokenize = {1: tokenize(query)}
    res = get_page_title(\
      map(lambda results: results[0], \
        get_topN_score_for_queries(query_tokenize, app.inverted_index_anchor, N=100, score=binary_similarity, base_dir='anchor')[1]))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    # return the selected pageRank
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc in wiki_ids:
        res.append(tuple([app.pageRank.get(doc, 'PageRank missing')]))
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    # return the selected pageView
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for doc in wiki_ids:
      res.append(tuple([app.pageViews.get(doc, 'PageView missing')]))
    # END SOLUTION
    return jsonify(res)


TUPLE_SIZE = 6       
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer
from contextlib import closing
import math 

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

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
    list_of_tokens =  [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]    
    return list_of_tokens


def get_page_title(wiki_ids):
  res = [tuple([id, app.wikiid_titles.get(id, 'Not Found!')]) for id in wiki_ids]
  return res


def get_term_total(index, base_dir='.'):
    
    term_total = {}
    counter = 0 
    for posting in index.posting_lists_iter(base_dir):
        w, post = posting
        for _, tf in post:
            term_total[w] = term_total.get(w, 0 ) + tf
            counter += 1
            if counter > 5:
                break
    return term_total


def read_posting_list(inverted, w, base_dir=''):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, base_dir)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list

def mysum(ls):
    sum_total =0
    for x in ls:
        sum_total += x
    return sum_total

def generate_query_tfidf_vector(query_to_search,index):
    epsilon = .0000001
    total_vocab_size = len(query_to_search)
    Q = np.zeros((total_vocab_size))
    term_vector = list(query_to_search)    
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys(): #avoid terms that do not appear in the index.               
            tf = counter[token]/len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]            
            idf = math.log((len(index.DS))/(df+epsilon),10) #smoothing
            
            try:
                ind = term_vector.index(token)
                Q[ind] = tf*idf                    
            except:
                pass
    return Q

def get_candidate_documents_and_scores(query_to_search,index, base_dir='.'):
    
    candidates = {}

    for term in np.unique(query_to_search):
        if term in index.df.keys():            
            list_of_doc = read_posting_list(index, term, base_dir)            
            normlized_tfidf = [(doc_id,(freq/index.DS[doc_id]['dl'])*math.log(len(index.DS)/index.df[term],10)) for doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf      

    return candidates

def generate_document_tfidf_matrix(query_to_search,index, base_dir='.'):

    total_vocab_size = len(query_to_search)
    candidates_scores = get_candidate_documents_and_scores(query_to_search,index,base_dir) #We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)
    
    D.index = unique_candidates
    D.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key    
        D.loc[doc_id][term] = tfidf

    return D

def get_top_n(sim_dict,N=3):
    """ 
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores 
   
    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3
    
    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    
    return sorted([(doc_id,round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]


def get_topN_score_for_queries(queries_to_search,index,N=3, score=None, base_dir='.'):
    """ 
    Generate a dictionary that gathers for every query its topN score.
    
    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows: 
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.    
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function. 
    
    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score). 
    """
    # YOUR CODE HERE
    results = {}

    for q_id, query in queries_to_search.items():
        Q = generate_query_tfidf_vector(query, index)
        D = generate_document_tfidf_matrix(query, index, base_dir)
        results[q_id] = get_top_n(score(D, Q), N)

    return results

def binary_similarity(d, q):
    d_bool = d.astype(bool)

    scores = d_bool.sum(axis=1)

    return scores.to_dict()

def cosine_similarity(D,Q):
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
    q_norm = np.linalg.norm(Q)
    d_norm = np.sqrt(np.square(D).sum(axis=1))
    dot_product = D.dot(Q)
    cos_sim = dot_product / (q_norm * d_norm)
    cos_sim_dict =  cos_sim.to_dict()
    return cos_sim_dict

def b_tf_cosine_similarity(queries_to_search, index, N=3, base_dir='.'):
    results = {}
    for q_id, query in queries_to_search.items():
      Q = generate_query_tfidf_vector(query, index)
      D = generate_document_tfidf_matrix(query, index, base_dir)

      q_norm = np.linalg.norm(Q)
      dot_product = D.dot(Q)
      cos_sim = dot_product / q_norm
      cos_sim_dict = {doc_id: score/index.DS[doc_id]['norm_mtf'] for doc_id, score in cos_sim.to_dict().items()}
      results[q_id] = get_top_n(cos_sim_dict, N)

    return results

import math
from itertools import chain
import time
# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25_from_index:
    """
    Best Match 25.    
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self,index,k1=1.5, b=0.75, base_dir=''):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DS)
        self.AVGDL = mysum(map(lambda ds: ds['dl'], index.DS.values()))/self.N
        self.base_dir = base_dir
        self.term_posting_list = {}

    def calc_idf(self,list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        
        Returns:
        -----------
        idf: dictionary of idf scores. As follows: 
                                                    key: term
                                                    value: bm25 idf score
        """        
        idf = {}        
        for term in list_of_tokens:            
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass                             
        return idf
        

    def search(self, queries,N=3, cosine=False):
        # YOUR CODE HERE
        score = 0.0
        quries_scores = {}
        if cosine:
          top_N = b_tf_cosine_similarity(queries, index=self.index, N=N, base_dir=self.base_dir)
        else:
          top_N = get_topN_score_for_queries(queries, score=binary_similarity, index=self.index, N=N, base_dir=self.base_dir)
        for query_id, res in top_N.items():
            rl_query = []
            self.idf = self.calc_idf(queries[query_id])
            for term in set(queries[query_id]):
                if term in self.index.term_total.keys():
                    self.term_posting_list[term] = \
                    dict(read_posting_list(self.index, term, self.base_dir))
                    rl_query.append(term)        
        docs_scores = {doc_id: self._score(rl_query, doc_id) for doc_id, _ in res}
        quries_scores[query_id] = docs_scores
        score = 0.0
        self.term_posting_list = {}
        rl_query = []
        return quries_scores

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.
        
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        
        Returns:
        -----------
        score: float, bm25 score.
        """        
        score = 0.0        
        doc_len = self.index.DS[doc_id]['dl']
             
        for term in query:
            if doc_id in self.term_posting_list[term].keys():            
                freq = self.term_posting_list[term][doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)
        return score

def merge_results(anchor_scores={}, title_scores={},body_scores={},title_weight=0.6,text_weight=0.15, anchor_weight = 0.40,N = 3):    
    # YOUR CODE HERE

    merged_results = {}
    for k in title_scores.keys():
        ts_dict = dict(title_scores.get(k, {}))
        bs_dict = dict(body_scores.get(k, {}))
        as_dict = dict(anchor_scores.get(k, {}))
        docs_keys = set(list(ts_dict.keys()) + list(bs_dict.keys()) + list(as_dict.keys()))
        merged_results[k] = sorted([(dk, as_dict.get(dk, 0)*anchor_weight + ts_dict.get(dk, 0)*title_weight + bs_dict.get(dk, 0)*text_weight) for dk in docs_keys],  key = lambda x: x[1],reverse=True)[:N]
    return merged_results

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
