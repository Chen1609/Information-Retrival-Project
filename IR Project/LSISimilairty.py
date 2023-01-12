from pageViews import PageViews
import numpy as np
import pandas as pd
from numpy import linalg as LA
from cosineSimilarity import Cosine_Similarity
import os


class LSI_Similarity:

    def __init__(self, index, k, path=os.path.join(os.getcwd(), 'lsi'), th=0.5):
        self.index = index
        self.k = k
        self.path = path
        self.th = th
    
    def get_posting_iter(self, index):
        """
        This function returning the iterator working with posting list.
        
        Parameters:
        ----------
        index: inverted index    
        """
        words, pls = zip(*index.posting_lists_iter())
        return words,pls
    
        
    def get_page_views_filterd_threashold(self, docs):
        pageViews_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
        docs_views = PageViews(pageViews_path, docs)
        docs_views_sorted = sorted(list(docs_views.items()), key=lambda x: x[1], reverse=True)
        maximum = docs_views_sorted[0][1]
        minimum = docs_views_sorted[len(docs_views)-1][1]
        docs_views_filter = filter(lambda x: (x[1]-minimum)/(maximum-minimum) > 0.005, docs_views_sorted)

        words, pls = self.get_posting_iter(self.index)
        return list(docs_views_filter)