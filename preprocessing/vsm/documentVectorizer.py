import numpy as np
import math
from ..indexing.index_api import IndexApi
from scipy import sparse

class DocumentVectorizer:
    def __init__(self, index_path: str, out_path):
        self.indexAPI = IndexApi(index_path)
        self.out_path = out_path

    def compute_tf_vector(self, doc_id: int) -> np.array:
        # Computes the tf vector for a single document
        vocab = self.indexAPI.get_sorted_vocabulary()
        T = len(vocab)
        tf_vector = np.empty(shape=(T,))
        for term_i, term in enumerate(vocab):
            postings_list = self.indexAPI.get_postings_list(term)
            tf = postings_list.get_term_frequency(doc_id)
            if tf == 0:
                tf_vector[term_i] = 0
            else:
                tf_vector[term_i] = 1 + math.log10(tf)
        return tf_vector
        
    def compute_doc_matrix(self) -> np.matrix:
        vocab = self.indexAPI.get_sorted_vocabulary()
        doc_ids = self.indexAPI.get_document_ids()
        T = len(vocab)
        N = len(doc_ids)
        doc_frequencies = np.empty(shape=(T, 1))
        # Compute document frequencies and initialize
        # matrix to N times idf vector
        for term_i, term in enumerate(vocab):
            postings_list = self.indexAPI.get_postings_list(term)
            df = postings_list.get_document_frequency()
            doc_frequencies[term_i] = math.log10(N/df)
        doc_matrix = np.tile(doc_frequencies, N)
        # For each doc compute term frequencies and multiply
        # tf vector element-wise with idf vector (= tf-idf)
        for doc_id in doc_ids:
            tf_vector = self.compute_tf_vector(doc_id)
            col = doc_matrix[:, doc_id-1]
            tf_idf_vector = np.multiply(col, tf_vector)
            doc_matrix[:, doc_id-1] = tf_idf_vector
        # Normalize vectors
        for doc_id in range(N):
            col = doc_matrix[:, doc_id]
            norm = np.linalg.norm(col, ord=2)
            if norm != 0:
                normalized = col/np.linalg.norm(col, ord=2)
                doc_matrix[:, doc_id] = normalized
        # Convert to sparse format
        sparse_matrix = sparse.csr_matrix(doc_matrix)
        sparse.save_npz("doc_vectors", sparse_matrix)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("inverted_index", "doc_vectors")
    vectorizer.compute_doc_matrix()