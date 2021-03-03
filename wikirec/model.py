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
from gensim.models import CoherenceModel

from wikirec import utils, topic_model


def get_topic_words(text_corpus, labels, num_topics=None, num_keywords=None):
    """
    Get top words within each topic for cluster models

    Parameters
    ----------
        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

        labels : list
            The labels assigned to topics

        num_topics : int (default=None)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=None)
            The number of keywords that should be extracted

    Returns
    -------
        topics, non_blank_topic_idxs : list and list
            Topic keywords and indexes of those that are not empty lists
    """
    if num_topics == None:
        num_topics = len(np.unique(labels))
    topics = ["" for _ in range(num_topics)]
    for i, c in enumerate(text_corpus):
        topics[labels[i]] += " " + " ".join(c)

    # Count the words that appear for a given topic label
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    word_counts = list(
        map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts)
    )

    topics = list(
        map(lambda x: list(map(lambda x: x[0], x[:num_keywords])), word_counts)
    )

    non_blank_topic_idxs = [i for i, t in enumerate(topics) if t != []]
    topics = [topics[i] for i in non_blank_topic_idxs]

    return topics, non_blank_topic_idxs


def get_coherence(model, text_corpus, num_topics=10, num_keywords=10, measure="c_v"):
    """
    Gets model coherence from gensim.models.coherencemodel

    Parameters
    ----------
        model : wikirec.topic_model.TopicModel
            A model trained on the given text corpus

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=10)
            The number of keywords that should be extracted

        measure : str (default=c_v)
            A gensim measure of coherence

    Returns
    -------
        coherence : float
            The coherence of the given model over the given texts
    """
    if model.method.lower() == "lda":
        cm = CoherenceModel(
            model=model.lda_model,
            texts=text_corpus,
            corpus=model.bow_corpus,
            dictionary=model.dirichlet_dict,
            coherence=measure,
        )
    else:
        topic_words = get_topic_words(
            text_corpus=text_corpus,
            labels=model.cluster_model.labels_,
            num_topics=num_topics,
            num_keywords=num_keywords,
        )[0]

        cm = CoherenceModel(
            topics=topic_words,
            texts=text_corpus,
            corpus=model.bow_corpus,
            dictionary=model.dirichlet_dict,
            coherence=measure,
        )

    coherence = cm.get_coherence()

    return coherence


def _order_and_subset_by_coherence(model, num_topics=10, num_keywords=10):
    """
    Orders topics based on their average coherence across the text corpus

    Parameters
    ----------
        model : wikirec.topic_model.TopicModel
            A model trained on the given text corpus

        num_topics : int (default=10)
            The number of categories for LDA and BERT based approaches

        num_keywords : int (default=10)
            The number of keywords that should be extracted

    Returns
    -------
        ordered_topic_words, selection_indexes: list of lists and list of lists
            Topics words ordered by average coherence and indexes by which they should be selected
    """
    # Derive average topics across texts for a given method
    if model.method == "lda":
        shown_topics = model.lda_model.show_topics(
            num_topics=num_topics, num_words=num_keywords, formatted=False
        )

        topic_words = [[word[0] for word in topic[1]] for topic in shown_topics]
        topic_corpus = model.lda_model.__getitem__(
            bow=model.bow_corpus, eps=0
        )  # cutoff probability to 0

        topics_per_response = [response for response in topic_corpus]
        flat_topic_coherences = [
            item for sublist in topics_per_response for item in sublist
        ]

        topic_averages = [
            (
                t,
                sum([t_c[1] for t_c in flat_topic_coherences if t_c[0] == t])
                / len(model.bow_corpus),
            )
            for t in range(num_topics)
        ]

    elif model.method == "bert" or model.method == "lda_bert":
        # The topics in cluster models are not guranteed to be the size of num_keywords
        topic_words, non_blank_topic_idxs = get_topic_words(
            text_corpus=model.text_corpus,
            labels=model.cluster_model.labels_,
            num_topics=num_topics,
            num_keywords=num_keywords,
        )

        # Create a dictionary of the assignment counts for the topics
        counts_dict = dict(Counter(model.cluster_model.labels_))
        counts_dict = {
            k: v for k, v in counts_dict.items() if k in non_blank_topic_idxs
        }
        keys_ordered = sorted([k for k in counts_dict.keys()])

        # Map to the range from 0 to the number of non-blank topics
        counts_dict_mapped = {i: counts_dict[k] for i, k in enumerate(keys_ordered)}

        # Derive the average assignment of the topics
        topic_averages = [
            (k, counts_dict_mapped[k] / sum(counts_dict_mapped.values()))
            for k in counts_dict_mapped.keys()
        ]

    # Order ids by the average coherence across the texts
    topic_ids_ordered = [
        tup[0] for tup in sorted(enumerate(topic_averages), key=lambda i: i[1][1])[::-1]
    ]
    ordered_topic_words = [topic_words[i] for i in topic_ids_ordered]

    ordered_topic_averages = [
        tup[1] for tup in sorted(topic_averages, key=lambda i: i[1])[::-1]
    ]
    ordered_topic_averages = [
        a / sum(ordered_topic_averages) for a in ordered_topic_averages
    ]  # normalize just in case

    # Create selection indexes for each topic given its average coherence and how many keywords are wanted
    selection_indexes = [
        list(range(int(math.floor(num_keywords * a))))
        if math.floor(num_keywords * a) > 0
        else [0]
        for i, a in enumerate(ordered_topic_averages)
    ]

    total_indexes = sum([len(i) for i in selection_indexes])
    s_i = 0
    while total_indexes < num_keywords:
        selection_indexes[s_i] = selection_indexes[s_i] + [
            selection_indexes[s_i][-1] + 1
        ]
        s_i += 1
        total_indexes += 1

    return ordered_topic_words, selection_indexes


def recommend(
    method="lda",
    inputs=None,
    texts_clean=None,
    text_corpus=None,
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
                frequency: a count of the most frequent words

                TFIDF: Term Frequency Inverse Document Frequency

                    - Allows for words within one text group to be compared to those of another
                    - Gives a better idea of what users specifically want from a given publication

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories
                    - These categories are then used to classify individual entries given the percent they fall into categories

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks
                    - Word classifications are then used to derive topics

                LDA_BERT: Latent Dirichlet Allocation with BERT embeddigs

                    - The combination of LDA and BERT via an autoencoder

        inputs : str or list (default=None)
            The name of an item or items of interest

        texts_clean : list (default=None)
            The texts from which recommendations can be made

        text_corpus : list, list of lists, or str
            The text corpus over which analysis should be done

            Note 1: generated using prepare_text_data

            Note 2: if a str is provided, then the data will be loaded from a path

        clean_texts : list
            Text strings that are formatted for cluster models

        n : int (default=10)
            The number of items to recommend

    Returns
    -------
        recommendations : list
            Those items that are most similar to the inputs
    """
    method = method.lower()

    valid_methods = ["lda", "bert", "lda_bert"]

    assert (
        method in valid_methods
    ), "The value for the 'method' argument is invalid. Please choose one of ".join(
        valid_methods
    )

    dictionary = corpora.Dictionary(text_corpus)

    bow_corpus = [dictionary.doc2bow(text) for text in text_corpus]

    index = similarities.MatrixSimilarity(topic_model.TopicModel(bow_corpus))

    books_checked = 0
    for i in range(len(texts_clean)):
        recommendation_scores = []
        if texts_clean[i][0] == inputs:
            lda_vectors = text_corpus[i]
            sims = index[lda_vectors]
            sims = list(enumerate(sims))
            for sim in sims:
                book_num = sim[0]
                recommendation_score = [texts_clean[book_num][0], sim[1]]
                recommendation_scores.append(recommendation_score)

            recommendation = sorted(
                recommendation_scores, key=lambda x: x[1], reverse=True
            )
            print("Your book's most prominent tokens are:")
            article_tokens = bow_corpus[i]
            sorted_tokens = sorted(article_tokens, key=lambda x: x[1], reverse=True)
            sorted_tokens_10 = sorted_tokens[:10]
            for i in range(len(sorted_tokens_10)):
                print(
                    'Word {} ("{}") appears {} time(s).'.format(
                        sorted_tokens_10[i][0],
                        dictionary[sorted_tokens_10[i][0]],
                        sorted_tokens_10[i][1],
                    )
                )
            print("-----")
            print("Your book's most prominant topic is:")
            print(
                get_topic_words(
                    labels=max(lda_vectors, key=lambda item: item[1])[0],
                    text_corpus=text_corpus,
                )
            )
            print("-----")
            print('Here are your recommendations for "{}":'.format(inputs))
            print(recommendation[1:11])

        else:
            books_checked += 1

        if books_checked == len(texts_clean):
            book_suggestions = []
            print('Sorry, but it looks like "{}" is not available.'.format(inputs))
            for x in range(len(texts_clean)):
                other_book = texts_clean[x][0]
                book_silimarity = round(utils.check_str_args(other_book, inputs), 2)
                similarity_score = [other_book, book_silimarity]
                book_suggestions.append(similarity_score)

            print("-----")
            ordered_suggestions = sorted(
                book_suggestions, key=lambda x: x[1], reverse=True
            )
            print("Were any of the following maybe what you were looking for?")
            print(ordered_suggestions[:10])
