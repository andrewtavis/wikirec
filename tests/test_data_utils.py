"""
Data Tests
----------
"""

import os
import spacy

from wikirec import data_utils


def test_input_conversion_dict():
    assert type(data_utils.input_conversion_dict()) == dict


def test__combine_tokens_to_str():
    texts = ["words", "to", "not", "be", "combined"]
    texts_lol = [["words", "to"], ["not"], ["be", "combined"]]
    ignore_words = ["not"]
    result_ignore_not = "words to be combined"
    result_not = "words to not be combined"

    assert (
        data_utils._combine_tokens_to_str(texts=texts, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert (
        data_utils._combine_tokens_to_str(texts=texts_lol, ignore_words=ignore_words)
        == result_ignore_not
    )

    assert (
        data_utils._combine_tokens_to_str(texts=texts, ignore_words=None) == result_not
    )

    assert (
        data_utils._combine_tokens_to_str(texts=texts_lol, ignore_words=None)
        == result_not
    )


def test_lemmatize():
    try:
        nlp = spacy.load("en")
    except:
        os.system("python -m spacy download {}".format("en"))
        nlp = spacy.load("en")
    assert data_utils.lemmatize([["better"], ["walking"], ["dogs"]], nlp=nlp) == [
        ["well"],
        ["walk"],
        ["dog"],
    ]
