import json

class IndexApi:
    def __init__(self, index_file_path: str):
        self.index_file_path = index_file_path
        with open(index_file_path, 'r') as index_file:
            self.dictionary = json.loads(index_file.read())
    
    def get_document_id_range(self) -> int:
        return 0

    def get_sorted_vocabulary(self) -> list[str]:
        return sorted(self.dictionary.keys())

    def get_postings_list(self, term: str) -> dict[int, list[int]]:
        return self.dictionary[term]

if __name__ == "__main__":
    api = IndexApi('dict.txt')
    vocab = api.get_sorted_vocabulary()
    for doc_id in range(0,100):
        