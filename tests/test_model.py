"""
Utilities Tests
---------------
"""

import numpy as np
import pytest

from wikirec import model

np.random.seed(42)


def test_gen_sim_matrix(text_corpus):
    bert_embeddings = model.gen_embeddings(
        method="bert",
        corpus=text_corpus,
        bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    )
    sim_matrix = model.gen_sim_matrix(
        method="bert", metric="euclidean", embeddings=bert_embeddings,
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="bert", metric="cosine", embeddings=bert_embeddings,
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    d2v_embeddings = model.gen_embeddings(method="doc2vec", corpus=text_corpus,)
    sim_matrix = model.gen_sim_matrix(
        method="doc2vec", metric="euclidean", embeddings=d2v_embeddings,
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    lda_embeddings = model.gen_embeddings(method="lda", corpus=text_corpus,)
    sim_matrix = model.gen_sim_matrix(
        method="lda", metric="cosine", embeddings=lda_embeddings,
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    sim_matrix = model.gen_sim_matrix(
        method="lda", metric="euclidean", embeddings=lda_embeddings,
    )

    assert sim_matrix == None

    tfidf_embeddings = model.gen_embeddings(method="tfidf", corpus=text_corpus,)

    sim_matrix = model.gen_sim_matrix(
        method="tfidf", metric="euclidean", embeddings=tfidf_embeddings,
    )

    assert type(sim_matrix) == np.ndarray
    assert len(sim_matrix) == len(text_corpus)

    with pytest.raises(ValueError):
        embeddings = model.gen_embeddings(  # pylint: disable=unused-variable
            method="Not a Method", corpus=text_corpus,
        )

    with pytest.raises(ValueError):
        sim_matrix = model.gen_sim_matrix(  # pylint: disable=unused-variable
            method="Not a Method", metric="euclidean", embeddings=tfidf_embeddings,
        )


def test_recommend(titles, sim_matrix_cosine, sim_matrix_euclidean):
    n = 5

    recs = model.recommend(
        inputs=titles[0],
        titles=titles,
        sim_matrix=sim_matrix_euclidean,
        n=n,
        metric="euclidean",
    )

    assert len(recs) == n
    assert recs[0][1] == min([r[1] for r in recs])

    recs = model.recommend(
        inputs=[titles[0], titles[1]],
        titles=titles,
        sim_matrix=sim_matrix_cosine,
        n=n,
        metric="cosine",
    )

    assert len(recs) == n
    assert recs[0][1] == max([r[1] for r in recs])
