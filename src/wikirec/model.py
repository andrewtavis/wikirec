"""
model
-----

Functions for modeling text corpuses and producing recommendations.

Contents:
    gen_embeddings,
    gen_sim_matrix,
    recommend,
    _wikilink_nn
"""

import json
import os
import random
import warnings
from collections import Counter, OrderedDict
from itertools import chain

import gensim
import numpy as np
from gensim import corpora, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamulticore import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tensorflow.keras import layers as tf_layers
from tensorflow.keras import models as tf_models
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from wikirec import utils


def gen_embeddings(
    method="bert",
    corpus=None,
    bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    path_to_json=None,
    path_to_embedding_model="wikilink_embedding_model",
    embedding_size=75,
    epochs=20,
    verbose=True,
    **kwargs,
):
    """
    Generates embeddings given a modeling method and text corpus.

    Parameters
    ----------
        method : str (default=bert)
            The modelling method.

            Options:
                BERT: Bidirectional Encoder Representations from Transformers

                    - Words embeddings are derived via Google Neural Networks.

                    - Embeddings are then used to derive similarities.

                Doc2vec : Document to Vector

                    - An entire document is converted to a vector.

                    - Based on word2vec, but maintains the document context.

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories.

                    - These categories are then used to classify individual entries given the percent they fall into categories.

                TFIDF: Term Frequency Inverse Document Frequency

                    - Word importance increases proportionally to the number of times a word appears in the document while being offset by the number of documents in the corpus that contain the word.

                    - These importances are then vectorized and used to relate documents.

                WikilinkNN: Wikilinks Neural Network

                    - Generate embeddings using a neural network trained on the connections between articles and their internal wikilinks.

        corpus : list of lists (default=None)
            The text corpus over which analysis should be done.

        bert_st_model : str (deafault=xlm-r-bert-base-nli-stsb-mean-tokens)
            The BERT model to use.

        path_to_json : str (default=None)
            The path to the parsed json file.

        path_to_embedding_model : str (default=wikilink_embedding_model)
            The name of the embedding model to load or create.

        embedding_size : int (default=75)
            The length of the embedding vectors between the articles and the links.

        epochs : int (default=20)
            The number of modeling iterations through the training dataset.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the model creation.

        **kwargs : keyword arguments
            Arguments correspoding to sentence_transformers.SentenceTransformer.encode, gensim.models.doc2vec.Doc2Vec, gensim.models.ldamulticore.LdaMulticore, or sklearn.feature_extraction.text.TfidfVectorizer.

    Returns
    -------
        embeddings : np.ndarray
            Embeddings to be used to create article-article similarity matrices.
    """
    method = method.lower()

    valid_methods = ["bert", "doc2vec", "lda", "tfidf", "wikilinknn"]

    if method not in valid_methods:
        raise ValueError(
            "The value for the 'method' argument is invalid. Please choose one of "
            + ", ".join(valid_methods)
        )

    if method == "bert":
        bert_model = SentenceTransformer(bert_st_model)

        return bert_model.encode(corpus, **kwargs)

    elif method == "doc2vec":
        tagged_data = [
            TaggedDocument(words=tc_i, tags=[i]) for i, tc_i in enumerate(corpus)
        ]

        v_size = kwargs.get("vector_size") if "vector_size" in kwargs else 100

        if float(gensim.__version__[0]) >= 4:
            model_d2v = Doc2Vec(documents=tagged_data, vector_size=v_size, **kwargs)

        else:
            model_d2v = Doc2Vec(vector_size=v_size, **kwargs)
            model_d2v.build_vocab(tagged_data)

            for _ in range(v_size):
                model_d2v.train(
                    documents=tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs,
                )

        embeddings = np.zeros((len(tagged_data), v_size))

        return [model_d2v.docvecs[i] for i, e in enumerate(embeddings)]

    elif method == "lda":
        if not isinstance(corpus[0], list):
            corpus = [c.split() for c in corpus]

        dictionary = corpora.Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in corpus]

        model_lda = LdaMulticore(corpus=bow_corpus, id2word=dictionary, **kwargs)

        return model_lda[bow_corpus]

    elif method == "tfidf":
        tfidfvectoriser = TfidfVectorizer(**kwargs)
        tfidfvectoriser.fit(corpus)

        return tfidfvectoriser.transform(corpus)

    elif method == "wikilinknn":
        if path_to_embedding_model[-3:] != ".h5":
            path_to_embedding_model += ".h5"

        model_name = path_to_embedding_model.split("/")[-1]

        if not os.path.isfile(path_to_embedding_model):
            print(f"Generating {model_name}.")
            return _wikilink_nn(
                path_to_json=path_to_json,
                path_to_embedding_model=path_to_embedding_model,
                embedding_size=embedding_size,
                epochs=epochs,
                verbose=verbose,
            )

        print(f"Loading {model_name}.")
        model = tf_models.load_model(path_to_embedding_model)
        layer = model.get_layer("article_embedding")
        weights = layer.get_weights()[0]

        return weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))


def gen_sim_matrix(
    method="bert", metric="cosine", embeddings=None,
):
    """
    Derives a similarity matrix from document embeddings.

    Parameters
    ----------
        method : str (default=bert)
            The modelling method.

            Options:
                BERT: Bidirectional Encoder Representations from Transformers

                    - Words embeddings are derived via Google Neural Networks.

                    - Embeddings are then used to derive similarities.

                Doc2vec : Document to Vector

                    - An entire document is converted to a vector.

                    - Based on word2vec, but maintains the document context.

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories.

                    - These categories are then used to classify individual entries given the percent they fall into categories.

                TFIDF: Term Frequency Inverse Document Frequency

                    - Word importance increases proportionally to the number of times a word appears in the document while being offset by the number of documents in the corpus that contain the word.

                    - These importances are then vectorized and used to relate documents.

                WikilinkNN: Wikilinks Neural Network

                    - Generate embeddings using a neural network trained on the connections between articles and their internal wikilinks.

        metric : str (default=cosine)
            The metric to be used when comparing vectorized corpus entries.

            Note: options include cosine and euclidean.

    Returns
    -------
        sim_matrix : gensim.interfaces.TransformedCorpus or numpy.ndarray
            The similarity sim_matrix for the corpus from the given model.
    """
    method = method.lower()

    valid_methods = ["bert", "doc2vec", "lda", "tfidf", "wikilinknn"]

    if method not in valid_methods:
        raise ValueError(
            "The value for the 'method' argument is invalid. Please choose one of "
            + ", ".join(valid_methods)
        )

    if method in ["bert", "doc2vec"]:
        if metric == "cosine":
            sim_matrix = cosine_similarity(embeddings)

        elif metric == "euclidean":
            sim_matrix = euclidean_distances(embeddings)

        return sim_matrix

    elif method == "lda":
        if metric == "cosine":
            sim_index = similarities.MatrixSimilarity(embeddings)
            sim_matrix = sim_index[embeddings]

            return sim_matrix

        elif metric == "euclidean":
            print(
                "Euclidean distance is not implemented for LDA modeling at this time. Please use 'cosine' for the metric argument."
            )
            return

    elif method in ["tfidf", "wikilinknn"]:
        if metric == "cosine" and method == "tfidf":
            sim_matrix = np.dot(  # pylint: disable=no-member
                embeddings, embeddings.T
            ).toarray()

        elif metric == "cosine" and method == "wikilinknn":
            sim_matrix = np.dot(embeddings, embeddings.T)  # pylint: disable=no-member

        elif metric == "euclidean":
            sim_matrix = euclidean_distances(embeddings)

        return sim_matrix


def recommend(
    inputs=None, ratings=None, titles=None, sim_matrix=None, metric="cosine", n=10,
):
    """
    Recommends similar items given an input or list of inputs of interest.

    Parameters
    ----------
        inputs : str or list (default=None)
            The name of an item or items of interest.

        ratings : list (default=None)
            A list of ratings that correspond to each input.

            Note: len(ratings) must equal len(inputs).

        titles : lists (default=None)
            The titles of the articles.

        sim_matrix : gensim.interfaces.TransformedCorpus or np.ndarray (default=None)
            The similarity sim_matrix for the corpus from the given model.

        n : int (default=10)
            The number of items to recommend.

        metric : str (default=cosine)
            The metric to be used when comparing vectorized corpus entries.

            Note: options include cosine and euclidean.

    Returns
    -------
        recommendations : list of lists
            Those items that are most similar to the inputs and their similarity scores
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    if ratings:
        if any(True for k in ratings if (k > 10) | (k < 0)):
            raise ValueError("Ratings must be between 0 and 10.")
        weights = np.divide(ratings, 10)

    first_input = True
    for r, inpt in enumerate(inputs):
        checked = 0
        num_missing = 0
        for i, t in enumerate(titles):
            if t == inpt:
                if first_input:
                    sims = sim_matrix[i]

                    first_input = False

                    if ratings:
                        sims = sims * weights[0]

                elif ratings:
                    sims = [
                        np.mean([r * s, weights[r] * sim_matrix[i][j]])
                        for j, s in enumerate(sims)
                    ]
                else:
                    sims = [
                        np.mean([r * s, sim_matrix[i][j]]) for j, s in enumerate(sims)
                    ]

            else:
                checked += 1
                if checked == len(titles):
                    num_missing += 1
                    print(f"{inpt} not available")
                    utils._check_str_args(arguments=inpt, valid_args=titles)

                    if num_missing == len(inputs):
                        ValueError(
                            "None of the provided inputs were found in the index. Please check them and reference Wikipedia for valid inputs via article names."
                        )

    titles_and_scores = [[t, sims[i]] for i, t in enumerate(titles)]

    if metric == "cosine":
        # Cosine similarities have been used (higher is better).
        recommendations = sorted(titles_and_scores, key=lambda x: x[1], reverse=True)
    elif metric == "euclidean":
        # Euclidean distances have been used (lower is better).
        recommendations = sorted(titles_and_scores, key=lambda x: x[1], reverse=False)

    recommendations = [r for r in recommendations if r[0] not in inputs][:n]

    return recommendations


def _wikilink_nn(
    path_to_json=None,
    path_to_embedding_model="wikilink_embedding_model",
    embedding_size=75,
    epochs=20,
    verbose=True,
):
    """
    Generates embeddings of wikilinks and articles by training a neural network.

    Parameters
    ----------
        path_to_json : str (default=None)
            The path to the parsed json file.

        path_to_embedding_model : str (default=wikilink_embedding_model)
            The name of the embedding model to load or create.

        embedding_size : int (default=75)
            The length of the embedding vectors between the articles and the links.

        epochs : int (default=20)
            The number of modeling iterations through the training dataset.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the model creation.

    Returns
    -------
        weights : np.array (len(corpus), embedding_size)
            The normalized embedding vectors for each of the articles.
    """
    if os.path.isfile(path_to_json):
        with open(path_to_json, "r") as fin:
            articles = [json.loads(l) for l in fin]
    else:
        raise FileNotFoundError("Need to parse json for articles.")

    # Find set of wikilinks for each article and convert to a flattened list.
    unique_wikilinks = list(chain(*[list(set(a[2])) for a in articles]))
    wikilinks = [link.lower() for link in unique_wikilinks]
    to_remove = [
        "hardcover",
        "paperback",
        "hardback",
        "e-book",
        "wikipedia:wikiproject books",
        "wikipedia:wikiproject novels",
    ]
    wikilinks = [link for link in wikilinks if link not in to_remove]

    # Limit to wikilinks that occur more than 4 times.
    wikilinks_counts = Counter(wikilinks)
    wikilinks_counts = sorted(
        wikilinks_counts.items(), key=lambda x: x[1], reverse=True
    )
    wikilinks_counts = OrderedDict(wikilinks_counts)
    desired_links = [t[0] for t in wikilinks_counts.items() if t[1] >= 4]

    # Map articles to their indices, and map links to indices as well.
    article_index = {a[0]: idx for idx, a in enumerate(articles)}
    link_index = {link: idx for idx, link in enumerate(desired_links)}

    # Create data from pairs of (article, wikilink) for training the neural network embedding.
    pairs = []
    disable = not verbose
    for article in tqdm(
        iterable=articles, desc="Article-link pairs made", disable=disable
    ):
        title = article[0]
        article_links = article[2]
        # Iterate through wikilinks in article.
        for link in article_links:
            # Add index of article and index of link to pairs.
            if link.lower() in desired_links:
                pairs.append((article_index[title], link_index[link.lower()]))

    pairs_set = set(pairs)

    # Neural network architecture.
    # Both inputs are 1-dimensional.
    article_input = tf_layers.Input(name="article", shape=[1])
    link_input = tf_layers.Input(name="link", shape=[1])

    # Embedding the article (shape will be (None, 1, embedding_size)).
    article_embedding = tf_layers.Embedding(
        name="article_embedding",
        input_dim=len(article_index),
        output_dim=embedding_size,
    )(article_input)

    # Embedding the link (shape will be (None, 1, embedding_size)).
    link_embedding = tf_layers.Embedding(
        name="link_embedding", input_dim=len(link_index), output_dim=embedding_size
    )(link_input)

    # Merge the layers with a dot product along the second axis
    # (shape will be (None, 1, 1)).
    merged = tf_layers.Dot(name="dot_product", normalize=True, axes=2)(
        [article_embedding, link_embedding]
    )

    # Reshape to be a single number (shape will be (None, 1)).
    merged = tf_layers.Reshape(target_shape=[1])(merged)

    model = tf_models.Model(inputs=[article_input, link_input], outputs=merged)
    model.compile(optimizer="Adam", loss="mse")

    # Function that creates a generator for training data.
    def _generate_batch(pairs, n_positive=embedding_size, negative_ratio=1.0):
        """
        Generate random positive and negative samples for training.
        """
        # Create empty array to hold batch.
        batch_size = n_positive * (1 + negative_ratio)
        batch = np.zeros((batch_size, 3))

        # Continue to yield samples.
        while True:
            # Randomly choose positive examples.
            for idx, (article_id, link_id) in enumerate(
                random.sample(pairs, n_positive)
            ):
                batch[idx, :] = (article_id, link_id, 1)
            idx += 1

            # Add negative examples until reach batch size.
            while idx < batch_size:

                # Random selection
                system_random = random.SystemRandom()
                random_article = system_random.randrange(len(article_index))
                random_link = system_random.randrange(len(link_index))

                # Check to make sure this is not a positive example.
                if (random_article, random_link) not in pairs_set:

                    # Add to batch and increment index.
                    batch[idx, :] = (random_article, random_link, 0)
                    idx += 1

            # Make sure to shuffle order.
            np.random.shuffle(batch)
            yield {"article": batch[:, 0], "link": batch[:, 1]}, batch[:, 2]

    n_positive = 1024
    # For testing purposes so that the sample is not larger.
    while n_positive >= len(pairs):
        n_positive -= 1

    gen = _generate_batch(pairs, n_positive, negative_ratio=2)

    if verbose == True:
        fit_verbose = 1
    else:
        fit_verbose = 0
    h = model.fit(  # pylint: disable=unused-variable
        gen,
        epochs=epochs,
        steps_per_epoch=len(pairs) // n_positive,
        verbose=fit_verbose,
    )

    # Save the model and extract embeddings.
    model_name = path_to_embedding_model.split("/")[-1]
    print(f"Saving {model_name}.")
    model.save(path_to_embedding_model)

    # Extract embeddings.
    article_layer = model.get_layer("article_embedding")
    article_weights = article_layer.get_weights()[0]

    # Normalize the weights to have norm of 1.
    article_weights = article_weights / np.linalg.norm(article_weights, axis=1).reshape(
        (-1, 1)
    )

    return article_weights
