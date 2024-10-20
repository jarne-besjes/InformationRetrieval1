import json
import os

class PostingsList:
    def __init__(self, postings: dict[str, int]):
        self._postings = postings
    
    def get_term_frequency(self, doc_id: int) -> int:
        doc_id_str = str(doc_id)
        if doc_id_str not in self._postings:
            return 0
        return self._postings[doc_id_str]
    
    def get_document_frequency(self):
        return sum(self._postings.values())

class IndexApi:
    def __init__(self, inverted_index_folder: str):
        self._index_file_path = inverted_index_folder
        self._posings_lists_file = open(os.path.join(inverted_index_folder, 'inverted_index.json') , 'r')
        with open(os.path.join(inverted_index_folder, 'dict.json'), 'r') as dict_file:
            self._dictionary = json.loads(dict_file.read())
        with open(os.path.join(inverted_index_folder, 'corpus_meta.txt'), 'r') as corpus_meta:
            self._doc_id_to_doc_name = json.loads(corpus_meta.read())

    # def get_sorted_vocabulary(self) -> list[str]:
    #     return sorted(self._postings_lists.keys())

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

if __name__ == "__main__":
    api = IndexApi('inverted_index')
    doc_ids = api.get_document_ids()

    postings = api.get_postings_list('crunch')
    tf = postings.get_term_frequency(1096)
    df = postings.get_document_frequency()
    print(tf, ' ', df)
