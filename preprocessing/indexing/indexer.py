from nltk.corpus.reader import documents
from nltk.tag.brill import Pos
from ..tokenizer import Tokenizer, TokenStream
import json
import bisect
import os
import os.path

class PostingsListsDict:
    def __init__(self):
        self.dictionary: dict[str, dict[int, list[int]]] = dict()
    
    def add_to_postings_list(self, token: str, pos: int, doc_id: int):
        if token not in self.dictionary.keys():
            self.dictionary[token] = dict()
        if doc_id not in self.dictionary[token].keys():
            self.dictionary[token][doc_id] = list()
        bisect.insort_left(self.dictionary[token][doc_id], pos)
        # self.dictionary[token][doc_id].append(pos)

def serialize_to_disk(dictionary: PostingsListsDict):
    # TODO: use something other than json as serialization format

    # We sort the term-keys and the document-keys in the outputted json for determinism and efficient search
    serialized = json.dumps(dictionary.dictionary, sort_keys=True)
    with open('./dict.txt', 'w+') as dict_file:
        dict_file.write(serialized)

class CorpusTokenizer:

    class Token:
        def __init__(self, token: str, pos: int, doc_id: int):
            self.token = token
            self.pos = pos
            self.doc_id = doc_id

    def __init__(self, documents_paths: list[str]):
        if len(documents_paths) == 0:
            raise ValueError("documents list cannot be empty")
        
        self.doc_i = 0
        self.doc_token_stream: TokenStream = Tokenizer.tokenize(documents_paths[self.doc_i])
        self.documents_paths = documents_paths

    # Returns True if this function managed to make another token available else False
    def _ensure_next_available(self) -> bool:
        if self.doc_token_stream.has_next():
            return True
        elif self.doc_i < len(self.documents_paths)-1:
            self.doc_i += 1
            self.doc_token_stream = Tokenizer.tokenize(self.documents_paths[self.doc_i])
            return self.doc_token_stream.has_next()
        else:
            return False

    def next(self) -> "Token | None":
        if not self._ensure_next_available():
            return None
        token = self.doc_token_stream.next()
        document_name = self.documents_paths[self.doc_i]
        return CorpusTokenizer.Token(token.token, token.pos, self.get_doc_id_from_filename(document_name))

    @staticmethod
    def get_doc_id_from_filename(filename: str) -> int:
        parts = filename.split('_')
        doc_id = parts[len(parts)-1][:-4] # strip .txt
        return int(doc_id)

def make_inverted_index_spimi(token_stream: CorpusTokenizer):
    MAX_TOKENS_BLOCK = 1_000_000
    postings_lists_dict = PostingsListsDict()
    cur_block_size = 0
    while (token := token_stream.next()) != None:
        postings_lists_dict.add_to_postings_list(token.token, token.pos, token.doc_id)
        cur_block_size += 1
        if cur_block_size > MAX_TOKENS_BLOCK:
            serialize_to_disk(postings_lists_dict)
            postings_lists_dict = PostingsListsDict()
            cur_block_size = 0
    if len(postings_lists_dict.dictionary.keys()) > 0:
        serialize_to_disk(postings_lists_dict) 

if __name__ == "__main__":
    import os
    print(print(os.getcwd()))
    dir_path = './full_docs_small'
    files = [dir_path + '/' + f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    token_stream = CorpusTokenizer(files)
    make_inverted_index_spimi(token_stream)

# while token_stream.has_next():
    #     postings_lists_dict = PostingsListsDict()
    #     tokens_processed = 0
    #     while tokens_processed < MAX_TOKENS_BATCH and token_stream.has_next():
    #         token = token_stream.next()
    #         postings_lists_dict.data[token.token].append((token.doc_id, token.pos))
    #         tokens_processed += 1
        
    #     serialize_to_disk(postings_lists_dict)

        # doc_i = 0
    # token_stream = tokenizer.Tokenizer.tokenize(documents[doc_i])
    # tokens_left = True
    # while tokens_left:
    #     postings_lists_dict = PostingsListsDict()
    #     tokens_processed = 0
    #     while tokens_processed < MAX_TOKENS_BATCH:
    #         # Get new token_stream if depleted
    #         if not token_stream.has_next():
    #             if doc_i != len(documents)-1:
    #                 doc_i += 1
    #                 token_stream = tokenizer.Tokenizer.tokenize(doc_i)
    #             else:
    #                 tokens_left = False
    #                 break
    #         token = token_stream.next()
    #         postings_lists_dict.data[token.token].append((doc_i, token.pos))
    #         tokens_processed += 1
    #     serialize_to_disk(postings_lists_dict)
    