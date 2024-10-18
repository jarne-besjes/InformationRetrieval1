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

    def compute_tf_vector(self, doc_id: int) -> np.array:
        # Computes the vector for a single document
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
        for term_i, term in enumerate(vocab):
            postings_list = self.indexAPI.get_postings_list(term)
            df = postings_list.get_document_frequency()
            doc_frequencies[term_i] = math.log10(N/df)
        doc_matrix = np.tile(doc_frequencies, N)
        print(doc_matrix)
        for doc_id in doc_ids:
            tf_vector = self.compute_tf_vector(doc_id)
            col = doc_matrix[:, doc_id-1]
            tf_idf_vector = np.multiply(col, tf_vector)
            doc_matrix[:, doc_id-1] = tf_idf_vector
        np.save(self.out_path, doc_matrix)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("inverted_index", "doc_vectors")
    vectorizer.compute_doc_matrix()