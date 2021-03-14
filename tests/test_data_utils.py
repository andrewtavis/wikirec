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
    texts = ["words", "to", "be", "combined"]
    texts_lol = [["words", "to"], ["be", "combined"]]
    result = "words to be combined"

    assert data_utils._combine_tokens_to_str(tokens=texts) == result

    assert data_utils._combine_tokens_to_str(tokens=texts_lol) == result


def test__lower_remove_unwanted():
    assert data_utils._lower_remove_unwanted(
        args=(
            ["Harry", "Potter", "25", "Zoo", "remove_please"],
            True,
            ["remove_please"],
            [],
        )
    ) == ["potter", "zoo"]

    assert data_utils._lower_remove_unwanted(
        args=(
            ["Harry", "Potter", "25", "Zoo", "remove_please"],
            False,
            ["remove_please"],
            [],
        )
    ) == ["harry", "potter", "zoo"]


def test__lemmatize():
    try:
        nlp = spacy.load("en")
    except:
        os.system("python -m spacy download {}".format("en"))
        nlp = spacy.load("en")
    assert data_utils._lemmatize([["better"], ["walking"], ["dogs"]], nlp=nlp) == [
        ["well"],
        ["walk"],
        ["dog"],
    ]


def test__subset_and_combine_tokens():
    assert data_utils._subset_and_combine_tokens(
        args=([0, ["here", "are", "some", "tokens", "and", "extras"]], 4)
    ) == [0, "here are some tokens"]
