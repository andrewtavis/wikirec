"""
Fixtures
--------
"""

import os
import json

import pytest
import numpy as np
from sentence_transformers import (
    SentenceTransformer,
)  # required or the import within wikirec.visuals will fail

import wikirec

from wikirec import data_utils
from wikirec.data_utils import input_conversion_dict
from wikirec.data_utils import download_wiki
from wikirec.data_utils import _process_article
from wikirec.data_utils import iterate_and_parse_file
from wikirec.data_utils import parse_to_ndjson
from wikirec.data_utils import _combine_tokens_to_str
from wikirec.data_utils import _clean_text_strings
from wikirec.data_utils import languages
from wikirec.data_utils import clean

from wikirec import model
from wikirec.model import gen_sim_matrix
from wikirec.model import recommend

from wikirec.languages import lem_abbr_dict
from wikirec.languages import stem_abbr_dict
from wikirec.languages import sw_abbr_dict

from wikirec.utils import _check_str_similarity
from wikirec.utils import _check_str_args
from wikirec.utils import graph_lda_topic_evals
from wikirec.utils import english_names_list

np.random.seed(42)

input_dir = "../wikirec/enwiki_dump"

files = data_utils.download_wiki(
    language="en", target_dir=input_dir, file_limit=1, dump_id=False
)

# To check that it finds already downloaded file
files = data_utils.download_wiki(
    language="en", target_dir=input_dir, file_limit=1, dump_id=False
)

output_path = "../wikirec/enwiki_books.ndjson"
partitions_dir = "../wikirec/enwiki_partitions"
limit = 10
data_utils.parse_to_ndjson(
    topic="books",
    output_path=output_path,
    input_dir=input_dir,
    partitions_dir=partitions_dir,
    limit=limit,
    delete_parsed_files=True,
    multicore=True,
    verbose=True,
)

dump_file_path = f"{input_dir}/{os.listdir(input_dir)[0]}"

parse_args = ("books", dump_file_path, partitions_dir, limit, True)
data_utils.iterate_and_parse_file(args=parse_args)

# Again to check that it skips the parse
data_utils.iterate_and_parse_file(args=parse_args)

with open(output_path, "r") as fin:
    books = [json.loads(l) for l in fin]

book_titles = [b[0] for b in books]
book_texts = [b[1] for b in books]

txts, tokens = data_utils.clean(
    texts=book_texts,
    min_freq=2,
    min_word_len=3,
    max_text_len=None,
    remove_names=True,
    sample_size=1,
    verbose=True,
)[:2]

smc = model.gen_sim_matrix(method="doc2vec", metric="cosine", corpus=txts)

sme = model.gen_sim_matrix(method="tfidf", metric="euclidean", corpus=txts)


@pytest.fixture(params=[book_titles])
def titles(request):
    return request.param


@pytest.fixture(params=[book_texts])
def texts(request):
    return request.param


@pytest.fixture(params=[txts])
def text_corpus(request):
    return request.param


@pytest.fixture(params=[tokens])
def token_corpus(request):
    return request.param


@pytest.fixture(params=[smc])
def sim_matrix_cosine(request):
    return request.param


@pytest.fixture(params=[sme])
def sim_matrix_euclidean(request):
    return request.param


os.system(f"rm -rf {input_dir}")
os.system(f"rm -rf {partitions_dir}")
os.system(f"rm -rf {output_path}")
