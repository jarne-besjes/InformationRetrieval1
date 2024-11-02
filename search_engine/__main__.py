"""
Information retrieval project 1

Authors: Jarne Besjes, Kobe De Broeck, Liam Leirs
"""
import os
import sys
import time
import argparse
import search_engine.preprocessing.indexing.indexer as indexer
from search_engine.search.FastCosine import QueryProcessor
import numpy as np
import pandas as pd
from tqdm import tqdm


argparser = argparse.ArgumentParser(description='Information retrieval project 1')
argparser.add_argument('--query', type=str, help='The query to search for')
argparser.add_argument('-n', type=int, default=10, help='The number of results to return')
argparser.add_argument('--no-index', action='store_true', help='Do not index the documents')
argparser.add_argument('--docs-folder', type=str, help='The location of the folder where all documents are stored', required=True)
argparser.add_argument('--mode', type=str, help='Select mode: search or bench', required=True)

if __name__ == "__main__":
    try:
        args = argparser.parse_args()
    except:
        argparser.print_help()
        sys.exit(1)

    query = args.query
    n = args.n
    docs_folder = args.docs_folder
    index_folder = docs_folder.rstrip("/") + "_index"
    mode = args.mode

    if mode == "bench":
        bench = True
    elif mode == "search":
        bench = False
        if query is None:
            print('Please provide a query', file=sys.stderr)
            sys.exit(1)
    else:
        print('Please provide a valid mode', file=sys.stderr)
        sys.exit(1)

    if not args.no_index:
        # Check if the inverted index folder exists and is not older than 1 day
        if not os.path.exists("./" + index_folder) :
            print('Indexing...', file=sys.stderr)
            #indexer.run_indexer(docs_folder)
            print('Indexing done', file=sys.stderr)

    if bench:
        # benchmark
        queryProcessor = QueryProcessor(index_folder)
        queries = pd.read_csv("queries.tsv", sep="\t")
        query_ids = queries["Query number"]

        precisions = np.zeros((2, len(queries)))
        recalls = np.zeros((2, len(queries)))

        expected_results = pd.read_csv("expected_results.csv")

        for i, (query_id, query) in tqdm(enumerate(queries.itertuples(index=False)), total = len(queries), desc="Running Benchmark..."):
            for ki, k in enumerate([3, 10]):
                doc_ids = queryProcessor.get_top_k_results(query, k)
                expected = expected_results[expected_results["Query_number"] == query_id]["doc_number"]
                relevant = 0
                docs_k = doc_ids[:k]
                for doc in expected:
                    if doc in docs_k:
                        relevant += 1
                precision = relevant / k
                recall = relevant / len(expected)

                precisions[ki, i] = precision
                recalls[ki, i] = recall

        for i in range(2):
            print(f"Mean precision at {3 if i == 0 else 10}: ", np.mean(precisions[i]))
            print(f"Mean recall at {3 if i == 0 else 10}: ", np.mean(recalls[i]))

    else:
        # search
        queryProcessor = QueryProcessor(index_folder)
        results = queryProcessor.get_top_k_results(query, n)
        print(results)
