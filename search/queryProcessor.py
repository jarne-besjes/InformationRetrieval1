import numpy as np
from scipy import sparse
from preprocessing.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from preprocessing.indexing.index_api import IndexApi
import math
import pandas as pd
from pathlib import Path

class QueryProcessor:
    def __init__(self, doc_vectors_path: str, index_path: str):
        doc_matrix = sparse.load_npz(doc_vectors_path)
        self.doc_matrix = doc_matrix.todense()
        self.indexAPI = IndexApi(index_path)

    def compute_query_vector(self, query: str) -> np.array:
        tokens = Tokenizer.tokenize(query, file_input=False).tokens
        unique = np.unique(tokens)
        vocab = self.indexAPI.get_sorted_vocabulary()
        doc_ids = self.indexAPI.get_document_ids()
        T = len(vocab)
        N = len(doc_ids)
        query_vector = np.zeros(shape=(1, T))
        for token in unique:
            postings_list = self.indexAPI.get_postings_list(str(token))
            df = postings_list.get_document_frequency()
            if df == 0:
                continue
            tf = tokens.count(str(token))
            max_tf = max(postings_list._postings.values())
            tf_weight = 0.5 + (0.5*tf)/(max_tf)
            index = vocab.index(str(token))
            query_vector[:, index] = math.log10(N/df)*tf_weight
        norm = np.linalg.norm(query_vector, ord=2)
        if norm == 0:
            return query_vector
        normalized = query_vector/norm
        return normalized

        
    def process_query(self, query: str, k: int):
        query_vector = self.compute_query_vector(query)
        scores = np.dot(query_vector, self.doc_matrix)
        scores = scores.A1
        best = np.argsort(scores)[-k:][::-1]
        if query == "what does gyrene mean":
            pass
        return [int(doc+1) for doc in best]
    
if __name__ == "__main__":
    print(Path.cwd())
    queries = pd.read_csv("queries.csv")
    query_ids = queries["Query number"]
    results = pd.read_csv("results.csv")
    queryProcessor = QueryProcessor("doc_vectors.npz", "inverted_index")
    correct = 0
    total = len(queries["Query"])
    for query_id, query in queries.itertuples(index=False):
        doc_ids = queryProcessor.process_query(query, 10)
        print(query, " found: ", doc_ids, end=" ")
        expected = results[results["Query_number"] == query_id]["doc_number"]
        found = False
        print("expected:", end=" ")
        for val in expected:
            print(str(val) + ", ", end=" ")
            if int(val) in doc_ids:
                found = True
                if found == True:
                    correct += 1
                    break
        print("\n")

    print((correct/total)*100)
        