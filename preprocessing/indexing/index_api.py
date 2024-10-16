import json
import os

class PostingsList:
    def __init__(self, postings: dict[int, list[int]]):
        self._postings = postings
    
    def get_term_frequency(self, doc_id: int):
        if doc_id not in self._postings:
            return 0
        return len(self._postings[doc_id])
    
    def get_document_frequency(self):
        return len(self._postings.keys())

class IndexApi:
    def __init__(self, inverted_index_folder: str):
        self._index_file_path = inverted_index_folder
        with open(os.path.join(inverted_index_folder, 'dict.txt') , 'r') as index_file:
            self._dictionary = json.loads(index_file.read())
        with open(os.path.join(inverted_index_folder, 'corpus_meta.txt'), 'r') as corpus_meta:
            self._doc_id_to_doc_name = json.loads(corpus_meta.read())

    def get_sorted_vocabulary(self) -> list[str]:
        return sorted(self._dictionary.keys())

    def get_postings_list(self, term: str) -> PostingsList:
        return PostingsList(self._dictionary[term])

    def get_document_ids(self) -> list[int]:
        return [int(doc_id) for doc_id in self._doc_id_to_doc_name.keys()]

if __name__ == "__main__":
    api = IndexApi('inverted_index')
    vocab = api.get_sorted_vocabulary()
    doc_ids = api.get_document_ids()
    for doc_id in doc_ids[:10]:
        for term_i, term in enumerate(vocab):
            postings_list = api.get_postings_list(term)
            tf = postings_list.get_term_frequency(doc_id)
            df = postings_list.get_document_frequency()
            print(f"tf(doc: {doc_id}, term: {term}) = {tf}")
            print(f"df(term: {term}) = {df}")
