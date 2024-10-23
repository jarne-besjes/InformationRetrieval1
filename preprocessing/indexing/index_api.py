import json
from math import sqrt
import os
import math
import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..tokenizer import Tokenizer

class PostingsList:
    def __init__(self, postings: dict[str, int]):
        self._postings = postings
    
    def get_term_frequency(self, doc_id: int) -> int:
        doc_id_str = str(doc_id)
        if doc_id_str not in self._postings:
            return 0
        return self._postings[doc_id_str]
    
    def get_document_frequency(self):
        return len(self._postings.keys())

class IndexApi:
    def __init__(self, inverted_index_folder: str):
        self._index_file_path = inverted_index_folder
        self._posings_lists_file = open(os.path.join(inverted_index_folder, 'inverted_index.json') , 'r')
        with open(os.path.join(inverted_index_folder, 'dict.json'), 'r') as dict_file:
            self._dictionary = json.loads(dict_file.read())
            self._sorted_terms = sorted(self._dictionary.keys())
        with open(os.path.join(inverted_index_folder, 'corpus_meta.txt'), 'r') as corpus_meta:
            self._doc_id_to_doc_name = json.loads(corpus_meta.read())

    # def get_sorted_vocabulary(self) -> list[str]:
    #     return sorted(self._postings_lists.keys())

    def get_sorted_vocabulary(self) -> list[str]:
        return self._sorted_terms

    def get_postings_list(self, term: str) -> PostingsList:
        postings_list_file_pos = self._dictionary.get(term)
        if postings_list_file_pos is None:
            return PostingsList(dict())
        self._posings_lists_file.seek(postings_list_file_pos)
        postings_list_str = self._posings_lists_file.readline()
        postrings_list = json.loads(postings_list_str)
        return PostingsList(postrings_list)

    def get_document_ids(self) -> list[int]:
        return [int(doc_id) for doc_id in self._doc_id_to_doc_name.keys()]

def calculate_doc_vector_lengths(inverted_index_folder_path: str, output_folder_path: str):
    api = IndexApi(inverted_index_folder_path)

    doc_ids = api.get_document_ids()

    print(f'min: {min(doc_ids)} max: {max(doc_ids)}, len: {len(doc_ids)}')

    scores = np.zeros(max(doc_ids)+1)
    for term in api._dictionary.keys():
        postings_list = api.get_postings_list(term)
        df = postings_list.get_document_frequency()
        idf_w = math.log10(len(scores)/df)
        for doc in postings_list._postings.keys():
            tf = postings_list._postings[doc]
            w = (1 + math.log10(tf)) * idf_w
            scores[int(doc)] += w**2
    scores = np.sqrt(scores)

    os.makedirs(output_folder_path, exist_ok=True)
    np.savetxt(os.path.join(output_folder_path, "doc_vec_lengths.txt"), scores)

def cosine_score(query: str, doc_vec_lengths: np.ndarray, api: IndexApi, weight_func) -> np.ndarray:
    tokens = Tokenizer.tokenize(query, file_input=False).tokens

    query_term_freqs = {}
    for token in tokens:
        if token not in query_term_freqs:
            query_term_freqs[token] = 0
        query_term_freqs[token] += 1
    
    doc_ids = api.get_document_ids()

    scores = np.zeros(max(doc_ids)+1)
    for term in query_term_freqs.keys():
        postings_list = api.get_postings_list(term)
        df = postings_list.get_document_frequency()
        for doc in postings_list._postings.keys():
            tf_d = postings_list._postings[doc]
            tf_q = query_term_freqs[term]
            w_d = weight_func(tf_d, df, len(doc_ids))
            w_q = weight_func(tf_q, df, len(doc_ids))
            scores[int(doc)] += w_d*w_q
    scores = np.divide(scores, doc_vec_lengths, out=np.zeros_like(scores), where=doc_vec_lengths != 0)
    best_docs = np.argsort(scores)
    best_docs = np.flip(best_docs)
    return best_docs

if __name__ == "__main__":
    # api = IndexApi('inverted_index')
    # doc_ids = api.get_document_ids()

    # for term in api._dictionary.keys():
    #     postings = api.get_postings_list(term)
    #     print(postings._postings)

    # postings = api.get_postings_list('crunch')
    # tf = postings.get_term_frequency(1096)
    # df = postings.get_document_frequency()
    # print(tf, ' ', df)

    # print(api.get_sorted_vocabulary())

    def weight_func(tf, df, corpus_size):
        return (1 + math.log10(tf)) * math.log10(corpus_size/df)
    
    api = IndexApi("big-inverted-index")

    import time
    now = time.time()
    
    #calculate_doc_vector_lengths("inverted_index", "vsm_helper_data")
    start_time = time.time()
    doc_vec_lengths = np.loadtxt("big_vsm_helper_data/doc_vec_lengths.txt")
    #query = "types of road hugger tires"
    query = "does xpress bet charge to deposit money in your account"
    ranking = cosine_score(query, doc_vec_lengths, api, weight_func)
    print(ranking[:10])
    print(np.where(ranking == 159065))
    print(np.where(ranking == 327023))

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    
