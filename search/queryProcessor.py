import numpy as np
from scipy import sparse
from preprocessing.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from preprocessing.indexing.index_api import IndexApi
import math

class QueryProcessor:
    def __init__(self, doc_vectors_path: str, index_path: str):
        doc_matrix = sparse.load_npz(doc_vectors_path)
        self.doc_matrix = doc_matrix.todense()
        self.indexAPI = IndexApi(index_path)

    def compute_query_vector(self, query: str) -> np.array:
        tokens = word_tokenize(query)
        unique = np.unique(tokens)
        vocab = self.indexAPI.get_sorted_vocabulary()
        doc_ids = self.indexAPI.get_document_ids()
        T = len(vocab)
        N = len(doc_ids)
        query_vector = np.zeros(shape=(1, T))
        for token in unique:
            try:
                postings_list = self.indexAPI.get_postings_list(token)
            except KeyError:
                continue
            df = postings_list.get_document_frequency()
            tf = tokens.count(token)
            tf_weight = 1 + math.log10(tf)
            index = vocab.index(token)
            query_vector[:, index] = math.log10(N/df)*tf_weight
        norm = np.linalg.norm(query_vector)
        normalized = query_vector/norm
        return normalized

        
    def process_query(self, query: str, k: int):
        query_vector = self.compute_query_vector(query)
        scores = np.dot(query_vector, self.doc_matrix)
        scores = scores.A1
        best = np.argsort(scores)[-k:][::-1]
        
        return [(int(doc+1), round(float(scores[doc]), 4)) for doc in best]
    
if __name__ == "__main__":
    queryProcessor = QueryProcessor("doc_vectors.npz", "inverted_index")
    print(queryProcessor.process_query("types of road hugger tires", 5))