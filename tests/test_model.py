"""
Utilities Tests
---------------
"""

import numpy as np
import pytest

from wikirec import model

np.random.seed(42)


def test_gen_sim_matrix(text_corpus, token_corpus):
    sim_matrix = model.gen_sim_matrix(
        method="bert", metric="euclidean", corpus=text_corpus
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="bert", metric="cosine", corpus=text_corpus
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="doc2vec", metric="euclidean", corpus=text_corpus
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="lda", metric="cosine", corpus=token_corpus
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="lda", metric="euclidean", corpus=token_corpus
    )

    assert sim_matrix == None

    sim_matrix = model.gen_sim_matrix(
        method="tfidf", metric="cosine", corpus=text_corpus
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    with pytest.raises(ValueError):
        sim_matrix = model.gen_sim_matrix(
            method="Not a Method", metric="cosine", corpus=None
        )


def test_recommend(titles, sim_matrix_cosine, sim_matrix_euclidean):
    n = 5

    recs = model.recommend(
        inputs=titles[0], titles=titles, sim_matrix=sim_matrix_euclidean, n=n,
    )

    assert len(recs) == n

    recs = model.recommend(
        inputs=[titles[0], titles[1]], titles=titles, sim_matrix=sim_matrix_cosine, n=n,
    )

    assert len(recs) == n
