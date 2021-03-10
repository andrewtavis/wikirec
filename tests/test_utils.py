"""
Utilities Tests
---------------
"""

from wikirec import utils


def test__check_str_similarity():
    assert utils._check_str_similarity("word", "word") == 1


def test__check_str_args():
    assert utils._check_str_args("word_0", ["word_0", "word_1"]) == "word_0"
    assert utils._check_str_args(["word_0", "word_1"], ["word_0", "word_1"]) == [
        "word_0",
        "word_1",
    ]
    assert utils._check_str_args("word_2", ["word_0", "word_1"]) == None


def test_graph_lda_topic_evals(monkeypatch, text_corpus):
    utils.graph_lda_topic_evals(
        corpus=text_corpus,
        num_topic_words=10,
        topic_nums_to_compare=[9, 10],
        metrics=True,
        verbose=True,
    )
