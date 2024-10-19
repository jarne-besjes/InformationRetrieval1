from operator import pos
from nltk.corpus.reader import documents
from nltk.tag.brill import Pos
from numpy import block
from ..tokenizer import Tokenizer, TokenStream
import json
import bisect
import os
import os.path
import ijson

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

    def serialize_to_disk(self, folder_output_path: str, n: int):
        # TODO: use something other than json as serialization format

        # We sort the term-keys and the document-keys in the outputted json for determinism and efficient search
        serialized = json.dumps(self.dictionary, sort_keys=True)
        os.makedirs(folder_output_path, exist_ok=True)
        with open(os.path.join(folder_output_path, f'dict_{n}.txt'), 'w+') as dict_file:
            dict_file.write(serialized)

class CorpusTokenizer:

    class Token:
        def __init__(self, token: str, pos: int, doc_id: int):
            self.token = token
            self.pos = pos
            self.doc_id = doc_id

    def __init__(self, corpus: "Corpus"):
        self.doc_ids = corpus.get_document_ids()
        self.doc_id_i = 0 # index in self.doc_ids
        self.doc_token_stream = None # filled in in _ensure_next_available
        self.corpus = corpus

    # Returns True if this function managed to make another token available else False
    def _ensure_next_available(self) -> bool:
        if self.doc_token_stream is not None and self.doc_token_stream.has_next():
            return True
        elif self.doc_id_i < len(self.doc_ids)-1:
            self.doc_token_stream = Tokenizer.tokenize(self.corpus.get_doc_path(self.doc_ids[self.doc_id_i]))
            self.doc_id_i += 1
            return self.doc_token_stream.has_next()
        else:
            return False

    def next(self) -> "Token | None":
        if not self._ensure_next_available():
            return None
        assert(self.doc_token_stream is not None)
        token = self.doc_token_stream.next()
        return CorpusTokenizer.Token(token.token, token.pos, self.doc_ids[self.doc_id_i])

class Corpus:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        document_names = [f for f in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, f))]
        if len(document_names) == 0:
            raise ValueError("documents list cannot be empty")
        self.doc_id_to_doc_name: dict[int, str] = dict()
        for doc in document_names:
            if not doc.endswith('.txt'):
                continue
            # Exploit the fact that the file names contain a unique id after the underscore
            parts = doc.split('_')
            doc_id = int(parts[len(parts)-1][:-4]) # strip .txt
            self.doc_id_to_doc_name[doc_id] = doc

    def get_document_ids(self) -> list[int]:
        return list(self.doc_id_to_doc_name.keys())
    
    def get_doc_name(self, doc_id: int):
        return self.doc_id_to_doc_name[doc_id]
    
    def get_doc_path(self, doc_id: int):
        return os.path.join(self.corpus_path, self.get_doc_name(doc_id))
    
    def serialize_to_disk(self, folder_output_path: str):
        os.makedirs(folder_output_path, exist_ok=True)
        with open(os.path.join(folder_output_path, 'corpus_meta.txt'), 'w+') as corpus_meta:
            corpus_meta.write(json.dumps(self.doc_id_to_doc_name))

class InvertedIndexGenerator:
    """
    :params: 
        corpus_path: path to folder of the documents in the corpus
    """
    def __init__(self, corpus_path: str):
        self.corpus = Corpus(corpus_path)
        self.token_stream = CorpusTokenizer(self.corpus)

    def generate_spimi(self, folder_output_path: str):
        # Have a clean-slate output folder
        import shutil
        shutil.rmtree(folder_output_path)

        # Generate inverted index
        MAX_TOKENS_BLOCK = 100_000

        postings_lists_dict = PostingsListsDict()
        block_n = 0
        
        def serialize_inverted_index():
            nonlocal postings_lists_dict
            nonlocal block_n
            postings_lists_dict.serialize_to_disk(os.path.join(folder_output_path, 'l0'), block_n)
            block_n += 1
        
        cur_block_size = 0
        while (token := self.token_stream.next()) != None:
            postings_lists_dict.add_to_postings_list(token.token, token.pos, token.doc_id)
            cur_block_size += 1
            if cur_block_size > MAX_TOKENS_BLOCK:
                serialize_inverted_index()
                postings_lists_dict = PostingsListsDict()
                cur_block_size = 0
        if len(postings_lists_dict.dictionary.keys()) > 0:
            serialize_inverted_index()
        
        self.merge_all_blocks(0, folder_output_path)

        # Write corpus meta data in a separate file
        self.corpus.serialize_to_disk(folder_output_path)
    
    def merge(self, block0_path: str, block1_path: str, output_path: str, n: int) -> str:
        output_file_name = os.path.join(output_path, f'block_{n}')
        with open(block0_path, 'r') as block0_f:
            with open(block1_path, 'r') as block1_f:
                os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
                with open(output_file_name, 'w+') as output_f:
                    postings_lists0 = ijson.kvitems(block0_f, '')
                    postings_lists1 = ijson.kvitems(block1_f, '')
                    list0 = next(postings_lists0, None) # list0[0] == term, list0[1] == postings_list
                    list1 = next(postings_lists1, None)

                    is_first = True

                    def write_postings_list(term, postings_list, output_file):
                        nonlocal is_first
                        to_write =  '' if is_first else ','
                        if is_first: is_first = False
                        to_write += json.dumps({term: postings_list})[1:-1] # remove { }
                        output_file.write(to_write + '\n')
                    
                    output_f.write('{')
                    while True:
                        if list0 is None:
                            # dump rest of block1
                            while list1 is not None:
                                write_postings_list(list1[0], list1[1], output_f)
                                list1 = next(postings_lists1, None)
                            break
                        if list1 is None:
                            # dump rest of block0
                            while list0 is not None:
                                write_postings_list(list0[0], list0[1], output_f)
                                list0 = next(postings_lists0, None)
                            break
                        if list0[0] < list1[0]:
                            write_postings_list(list0[0], list0[1], output_f)
                            list0 = next(postings_lists0, None)
                        elif list1[0] < list0[0]:
                            write_postings_list(list1[0], list1[1], output_f)
                            list1 = next(postings_lists1, None)
                        else:
                            # t1 == t0 => merge lists
                            import copy
                            merged = copy.deepcopy(list0[1])
                            for d1, positions in list1[1].items():
                                if d1 in merged.keys():
                                    merged[d1] += positions
                                else:
                                    merged[d1] = positions
                            write_postings_list(list0[0], merged, output_f)
                            list0 = next(postings_lists0, None)
                            list1 = next(postings_lists1, None)
                    output_f.write('}')
                return output_file_name

    def merge_all_blocks(self, level: int, output_path: str):
        blocks_path = os.path.join(output_path, f'l{level}')
        blocks = [os.path.join(blocks_path, f) for f in os.listdir(blocks_path) if os.path.isfile(os.path.join(blocks_path, f))]
        
        if len(blocks) == 1:
            # Everything merged, we are done
            return

        # Create an empty dummy block to have an even number of blocks for the merge step
        if len(blocks) % 2 != 0:
            dummy_path = os.path.join(blocks_path, f'dummy_block')
            with open(dummy_path, 'w+') as dummy_file:
                dummy_file.write('{}')
                blocks.append(dummy_path)
                
        merged_output_path = os.path.join(output_path, f'l{level+1}')
        for i in range(0, len(blocks), 2):
            self.merge(blocks[i], blocks[i+1], merged_output_path, i)
        self.merge_all_blocks(level+1, output_path)

if __name__ == "__main__":
    index_gen = InvertedIndexGenerator(corpus_path='./full_docs_small')
    index_gen.generate_spimi(folder_output_path='inverted_index')

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
    