"""
Fixtures
--------
"""

import json
import os

import numpy as np
import pytest
from sentence_transformers import (
    SentenceTransformer,
)  # pylint: disable=unused-import; required or the import within wikirec.visuals will fail
from wikirec import data_utils, model

np.random.seed(42)

input_dir = "./test_files"

language = "en"

files = data_utils.download_wiki(
    language=language, target_dir=input_dir, file_limit=1, dump_id=False
)

# To check that it finds already downloaded file.
files = data_utils.download_wiki(
    language=language, target_dir=input_dir, file_limit=1, dump_id=False
)

output_path = "./test_files/enwiki_books.ndjson"
partitions_dir = "./test_files/enwiki_partitions"
limit = 10
data_utils.parse_to_ndjson(
    topics="books",
    output_path=output_path,
    input_dir=input_dir,
    partitions_dir=partitions_dir,
    limit=limit,
    delete_parsed_files=True,
    multicore=True,
    verbose=True,
)

dump_file_path = f"{input_dir}/{os.listdir(input_dir)[0]}"

parse_args = ("books", language, dump_file_path, partitions_dir, limit, True)
data_utils.iterate_and_parse_file(args=parse_args)

# Again to check that it skips the parse.
data_utils.iterate_and_parse_file(args=parse_args)

with open(output_path, "r") as f:
    books = [json.loads(l) for l in f]

book_titles = [b[0] for b in books]
book_texts = [b[1] for b in books]

txts = data_utils.clean(
    texts=book_texts,
    min_token_freq=2,
    min_token_len=3,
    min_tokens=0,
    max_token_index=-1,
    remove_names=True,
    sample_size=1,
    verbose=True,
)[0]

d2v_embeddings = model.gen_embeddings(method="doc2vec", corpus=txts)
sm_c = model.gen_sim_matrix(
    method="doc2vec", metric="cosine", embeddings=d2v_embeddings
)

tfidf_embeddings = model.gen_embeddings(method="tfidf", corpus=txts)
sm_e = model.gen_sim_matrix(
    method="tfidf", metric="euclidean", embeddings=tfidf_embeddings
)


@pytest.fixture(params=[book_titles])
def titles(request):
    return request.param


@pytest.fixture(params=[book_texts])
def texts(request):
    return request.param


@pytest.fixture(params=[txts])
def text_corpus(request):
    return request.param


@pytest.fixture(params=[sm_c])
def sim_matrix_cosine(request):
    return request.param


@pytest.fixture(params=[sm_e])
def sim_matrix_euclidean(request):
    return request.param


os.system(f"rm -rf {input_dir}")
os.system(f"rm -rf {partitions_dir}")
os.system(f"rm -rf {output_path}")
