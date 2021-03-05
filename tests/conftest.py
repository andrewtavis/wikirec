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

files = data_utils.download_wiki(
    language="en", target_dir="../wikirec/enwiki_dump", file_limit=1, dump_id=False
)

data_utils.parse_to_ndjson(
    topic="books",
    output_path="../wikirec/enwiki_books.ndjson",
    input_dir="../wikirec/enwiki_dump",
    partitions_dir="../wikirec/enwiki_partitions",
    limit=10,
    delete_parsed_files=True,
    multicore=True,
    verbose=True,
)

with open("../wikirec/enwiki_books.ndjson", "r") as fin:
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


os.system("rm -rf ../wikirec/enwiki_dump")
os.system("rm -rf ../wikirec/enwiki_books.ndjson")
