import numpy as np
import math
from ..indexing.index_api import IndexApi

class DocumentVectorizer:
    def __init__(self, index_path: str, out_path):
        self.indexAPI = IndexApi(index_path)
        self.out_path = out_path

    def tf_idf_weight(self, tf: int, df: int, N: int) -> float:
        # Returns the tf-idf weight of a term
        if tf == 0:
            return 0
        tf_weight = 1 + math.log10(tf)
        idf_weight = math.log10(N/df)
        return tf_weight * idf_weight

    def compute_doc_vector(self, doc_id: int, N: int) -> np.array:
        # Computes the vector for a single document
        vocab = self.indexAPI.get_sorted_vocabulary()
        T = len(vocab)
        doc_vector = np.empty(shape=(T, 1))
        for term_i, term in enumerate(vocab):
            postings_list = self.indexAPI.get_postings_list(term)
            tf = postings_list.get_term_frequency(doc_id)
            df = postings_list.get_document_frequency()
            tf_idf_weight = self.tf_idf_weight(tf, df, N)
            doc_vector[term_i,] = tf_idf_weight
        return doc_vector
        

    def compute_doc_matrix(self) -> np.matrix:
        vocab = self.indexAPI.get_sorted_vocabulary()
        doc_ids = self.indexAPI.get_document_ids()
        T = len(vocab)
        N = len(doc_ids)
        doc_matrix = np.empty(shape=(T, N))
        for doc_id in doc_ids:
            doc_vector = self.compute_doc_vector(doc_id, N)
            doc_matrix[:, doc_id-1] = doc_vector[:, 0]
        np.save(self.out_path, doc_matrix)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("inverted_index", "doc_vectors")
    vectorizer.compute_doc_matrix()