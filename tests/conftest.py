"""
Fixtures
--------
"""

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
from wikirec.data_utils import _iterate_and_parse_file
from wikirec.data_utils import parse_to_ndjson
from wikirec.data_utils import _combine_tokens_to_str
from wikirec.data_utils import _clean_text_strings
from wikirec.data_utils import clean

from wikirec import model
from wikirec.model import gen_sim_matrix
from wikirec.model import recommend

from wikirec import utils
from wikirec.utils import _check_str_similarity
from wikirec.utils import _check_str_args
from wikirec.utils import graph_lda_topic_evals
from wikirec.utils import english_names_list

np.random.seed(42)


@pytest.fixture(params=[])
def fixture(request):
    return request.param
