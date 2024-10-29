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

if __name__ == "__main__":
    queries = pd.read_csv("queries.tsv", sep="\t")
    query_ids = queries["Query number"]
    results = pd.read_csv("results.csv")
    queryProcessor = QueryProcessor("big-inverted-index")
    # queryProcessor.compute_doc_lengths("big-lengths")
    # queryProcessor.lengths = np.load("big-lengths.npy")
    max_k = 20
    precision_at_k = np.zeros((max_k, len(queries)))
    recall_at_k = np.zeros((max_k, len(queries)))
    with open("output.txt", "w") as file:
        for i, (query_id, query) in tqdm(enumerate(queries.itertuples(index=False)), total = len(queries), desc="Processing queries"):
            doc_ids = queryProcessor.get_top_k_results(query, max_k)
            file.write("Got: " + str(doc_ids))
            expected = results[results["Query_number"] == query_id]["doc_number"]
            file.write(" Expected: " + str([val for val in expected]) + "\n")
            for k in range(1, max_k+1):
                relevant = 0
                docs_k = doc_ids[:k]
                for doc in expected:
                    if doc in docs_k:
                        relevant += 1
                precision_at_k[k-1, i] = relevant / k
                recall_at_k[k-1, i] = relevant / len(expected)

        mean_precision_at_k = np.mean(precision_at_k, axis=1)
        mean_recall_at_k = np.mean(recall_at_k, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_recall_at_k, mean_precision_at_k, marker='o', color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()
    plt.show()