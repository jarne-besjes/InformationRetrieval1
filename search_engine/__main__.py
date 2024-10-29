"""
Information retrieval project 1

Authors: Jarne Besjes, Kobe De Broeck, Liam Leirs
"""
import os
import sys
import time
import argparse
import search_engine.preprocessing.indexing.indexer as indexer

argparser = argparse.ArgumentParser(description='Information retrieval project 1')
argparser.add_argument('--query', type=str, help='The query to search for', required=True)
argparser.add_argument('-n', type=int, default=10, help='The number of results to return')
argparser.add_argument('--no-index', action='store_true', help='Do not index the documents')
argparser.add_argument('--docs-folder', type=str, help='The location of the folder where all documents are stored', required=True)

if __name__ == "__main__":
    try:
        args = argparser.parse_args()
    except:
        argparser.print_help()
        sys.exit(1)

    query = args.query
    n = args.n
    docs_folders = args.docs_folder

    if not args.no_index:
        # Check if the inverted index folder exists and is not older than 1 day
        if not os.path.exists('inverted_index') or (time.time() - os.path.getmtime('inverted_index')) > 86400:
            print('Indexing...', file=sys.stderr)
            indexer.run_indexer()
            print('Indexing done', file=sys.stderr)

    # vectorize

    # search