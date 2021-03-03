"""
model
-----

Functions for modeling text corpuses and producing recommendations

Contents:
    get_topic_words,
    get_coherence,
    _order_and_subset_by_coherence,
    recommend
"""

from collections import Counter
import math

import numpy as np

from gensim import corpora, models, similarities
from gensim.models import LdaModel, CoherenceModel

from wikirec import utils


def recommend(
    method="lda",
    inputs=None,
    texts_clean=None,
    text_corpus=None,
    titles=None,
    clean_texts=None,
    n=10,
):
    """
    Recommends similar items given an input or list of inputs of interest

    Parameters
    ----------
        method : str (default=lda)
            The modelling method

            Options:
                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories
                    - These categories are then used to classify individual entries given the percent they fall into categories

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks
                    - Word classifications are then used to derive similarities

        inputs : str or list (default=None)
            The name of an item or items of interest

        text_corpus : list or list of lists
            The text corpus over which analysis should be done

        titles : lists (default=None)
            The titles of the articles

        n : int (default=10)
            The number of items to recommend

    Returns
    -------
        recommendations : list
            Those items that are most similar to the inputs
    """
    method = method.lower()

    valid_methods = ["lda", "bert"]

    assert (
        method in valid_methods
    ), "The value for the 'method' argument is invalid. Please choose one of ".join(
        valid_methods
    )

    dictionary = corpora.Dictionary(text_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in text_corpus]

    sim_index = similarities.MatrixSimilarity(bow_corpus)

    books_checked = 0
    for i in range(len(text_corpus)):
        recommendation_scores = []
        if titles[i] == inputs:
            lda_vectors = text_corpus[i]
            sims = sim_index[lda_vectors]
            sims = list(enumerate(sims))
            for sim in sims:
                book_num = sim[0]
                score = [texts_clean[book_num][0], sim[1]]
                recommendation_scores.append(score)

            recommendation = sorted(
                recommendation_scores, key=lambda x: x[1], reverse=True
            )
            return recommendation[1:n]

        else:
            books_checked += 1

        if books_checked == len(texts_clean):
            print(f"{inputs} not available")
            utils._check_str_args(arguments=inputs, valid_args=titles)
