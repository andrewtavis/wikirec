"""
model
-----

Functions for modeling text corpuses and producing recommendations

Contents:
    derive_similarities,
    recommend
"""

from collections import Counter
import math

import numpy as np

from gensim import corpora, models, similarities
from gensim.models import LdaModel, CoherenceModel

from wikirec import utils


def derive_similarities(
    method="lda", num_topics=10, text_corpus=None,
):
    """
    Derives similarities between the entries in the text corpus

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

        num_topics : int (default=10)
            The number of topics for LDA models

        text_corpus : list or list of lists (default=None)
            The text corpus over which analysis should be done

    Returns
    -------
        model : gensim.models.LdaModel or BERT
            The model with which recommendations should be made

        sim_index : gensim.similarities.docsim.MatrixSimilarity or BERT
            An index of similarities for all items

        vectors : gensim.interfaces.TransformedCorpus or BERT
            The similarity vectors for the corpus from the given model
    """
    method = method.lower()

    valid_methods = ["lda", "bert"]

    assert (
        method in valid_methods
    ), "The value for the 'method' argument is invalid. Please choose one of ".join(
        valid_methods
    )

    if method == "lda":
        dictionary = corpora.Dictionary(text_corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in text_corpus]

        model = LdaModel(
            bow_corpus,
            num_topics=num_topics,
            random_state=42,
            update_every=1,
            passes=10,
            id2word=dictionary,
        )

        sim_index = similarities.MatrixSimilarity(model[bow_corpus])

        vectors = model[bow_corpus]

    return model, sim_index, vectors


def recommend(
    inputs=None, model=None, sim_index=None, vectors=None, titles=None, n=10,
):
    """
    Recommends similar items given an input or list of inputs of interest

    Parameters
    ----------
        inputs : str or list (default=None)
            The name of an item or items of interest

        model : gensim.models.LdaModel or BERT
            The model with which recommendations should be made

        sim_index : gensim.similarities.docsim.MatrixSimilarity or BERT
            An index of similarities for all items

        vectors : gensim.interfaces.TransformedCorpus or BERT
            The similarity vectors for the corpus from the given model

        titles : lists (default=None)
            The titles of the articles

        n : int (default=10)
            The number of items to recommend

    Returns
    -------
        recommendations : list of lists
            Those items that are most similar to the inputs and their similarity scores
    """
    if type(inputs) == str:
        inputs = [inputs]

    sims = None
    for inpt in inputs:
        checked = 0
        for i in range(len(titles)):
            if titles[i] == inpt:
                if sims is None:
                    sims = sim_index[vectors[i]]
                else:
                    sims = [
                        np.mean([sims[j], sim_index[vectors[i]][j]])
                        for j in range(len(sims))
                    ]

            else:
                checked += 1
                if checked == len(titles):
                    print(f"{inputs} not available")
                    utils._check_str_args(arguments=inputs, valid_args=titles)

    titles_and_scores = [[titles[i], sims[i]] for i in range(len(titles))]

    recommendations = sorted(titles_and_scores, key=lambda x: x[1], reverse=True)
    recommendations = [r for r in recommendations if r[0] not in inputs][:n]

    return recommendations
