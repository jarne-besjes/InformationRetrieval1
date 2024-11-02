from search_engine.preprocessing.indexing.index_api import IndexApi
from search_engine.preprocessing.tokenizer import Tokenizer
import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class QueryProcessor:
    def __init__(self, index_path: str):
        self.indexAPI = IndexApi(index_path)
        self.lengths = None

    def compute_doc_lengths(self, out_path: str):
        # Precompute doc lengths
        vocab = self.indexAPI.get_sorted_vocabulary()
        doc_ids = self.indexAPI.get_document_ids()
        N = len(doc_ids)
        lengths = np.zeros(N)
        for term in tqdm(vocab):
            postings_list = self.indexAPI.get_postings_list(term)
            idf = np.log10(N / postings_list.get_document_frequency())
            for doc_id, tf in postings_list._postings.items():
                tf_d = 1 + math.log10(tf)
                w_d = tf_d * idf
                lengths[int(doc_id)-1] += (w_d ** 2)
        lengths = np.sqrt(lengths)
        np.save(out_path, lengths)

    def compute_similarities(self, query_tokens: list[str]):
        # Compute cosine similarities using the fast cosine algorithm
        doc_ids = self.indexAPI.get_document_ids()
        N = len(doc_ids)
        scores = np.zeros(N)
        lengths = self.lengths
        unique = np.unique(query_tokens)
        query_weights = np.zeros(len(unique))

        # Calculate score for each term in query
        for i, term in enumerate(unique):
            postings_list = self.indexAPI.get_postings_list(term)
            df = postings_list.get_document_frequency()
            if df == 0:
                continue
            idf = math.log10(N/df)
            tf_q = 1 + math.log10(query_tokens.count(term))
            w_q = tf_q * idf
            query_weights[i] = w_q
            for doc_id, tf in postings_list._postings.items():
                tf_d = 1 + math.log10(tf)
                w_d = tf_d * idf
                scores[int(doc_id)-1] += w_d * w_q
        
        # Normalize scores by doc and query norm
        # for doc_id in doc_ids:
        #     if lengths[int(doc_id)-1] > 0:
        #         scores[int(doc_id)-1] /= lengths[doc_id-1]
        
        # query_norm = np.linalg.norm(query_weights)
        
        # if query_norm > 0:
        #     scores /= query_norm
        
        return scores
    
    def get_top_k_results(self, query: str, k: int):
        tokens = Tokenizer.tokenize(query, file_input=False).tokens
        scores = self.compute_similarities(tokens)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [int(doc + 1) for doc in top_k_indices]

def average_x_at_k(retrieved, relevant, k, precision=True):
    # Helper function to calculate AP@k and AR@k
    count = 0
    sum = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
            if precision:
                sum += count / i
            else:
                sum += count / len(retrieved)
    return sum / count if count > 0 else 0

def max_at_k(queries, results, processor, k, precision=True):
    # Function to calculate MAP@k and MAR@k
    averages = []
    for query_id, query in tqdm(queries.itertuples(index=False), total=len(queries)):
        retrieved = processor.get_top_k_results(query, k)
        relevant = results[results["Query_number"] == query_id]["doc_number"].to_list()
        average = average_x_at_k(retrieved, relevant, k, precision)
        averages.append(average)
    return np.mean(averages)

if __name__ == "__main__":
    queries = pd.read_csv("queries.tsv", sep="\t")
    query_ids = queries["Query number"]
    results = pd.read_csv("results.csv")
    queryProcessor = QueryProcessor("big-inverted-index")
    # queryProcessor.compute_doc_lengths("big-lengths")
    # queryProcessor.lengths = np.load("big-lengths.npy")
    maps = []
    mars = []
    for k in tqdm(range(1, 11)):
        map = max_at_k(queries, results, queryProcessor, k, precision=True)
        mar = max_at_k(queries, results, queryProcessor, k, precision=False)
        maps.append(map)
        mars.append(mars)
    plt.figure(figsize=(10, 6))
    plt.plot(maps, mars, marker='o', color='b')
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()