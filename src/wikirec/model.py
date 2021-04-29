"""
model
-----

Functions for modeling text corpuses and producing recommendations.

Contents:
    gen_embeddings,
    gen_sim_matrix,
    recommend
"""

import warnings

import gensim
import numpy as np
from gensim import corpora, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.ldamulticore import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from wikirec import utils

import os
import json
import random
from itertools import chain
from collections import Counter, OrderedDict
from keras.models import load_model
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model


def gen_embeddings(
    method="lda",
    corpus=None,
    bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
    **kwargs,
):
    """
    Generates embeddings given a modeling method and text corpus.

    Parameters
    ----------
        method : str (default=lda)
            The modelling method

            Options:
                BERT: Bidirectional Encoder Representations from Transformers

                    - Words embeddings are derived via Google Neural Networks

                    - Embeddings are then used to derive similarities

                Doc2vec : Document to Vector

                    - An entire document is converted to a vector

                    - Based on word2vec, but maintains the document context

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories

                    - These categories are then used to classify individual entries given the percent they fall into categories

                TFIDF: Term Frequency Inverse Document Frequency

                    - Word importance increases proportionally to the number of times a word appears in the document while being offset by the number of documents in the corpus that contain the word

                    - These importances are then vectorized and used to relate documents
                    
                Wikilinks
                    
                    - Generate an embedding using a neural network trained on the connections between articles and the internal wikilinks

        corpus : list of lists (default=None)
            The text corpus over which analysis should be done

        bert_st_model : str (deafault=xlm-r-bert-base-nli-stsb-mean-tokens)
            The BERT model to use

        **kwargs : keyword arguments
            Arguments correspoding to sentence_transformers.SentenceTransformer.encode, gensim.models.doc2vec.Doc2Vec, gensim.models.ldamulticore.LdaMulticore, or sklearn.feature_extraction.text.TfidfVectorizer

    Returns
    -------
        embeddings :
            Embeddings to be used to create article-article similarity matrices
    """
    method = method.lower()

    valid_methods = ["bert", "doc2vec", "lda", "tfidf"]

    if method not in valid_methods:
        raise ValueError(
            "The value for the 'method' argument is invalid. Please choose one of ".join(
                valid_methods
            )
        )

    if method == "bert":
        bert_model = SentenceTransformer(bert_st_model)

        embeddings = bert_model.encode(corpus, **kwargs)

        return embeddings

    elif method == "doc2vec":
        tagged_data = [
            TaggedDocument(words=tc_i, tags=[i]) for i, tc_i in enumerate(corpus)
        ]

        v_size = kwargs.get("vector_size") if "vector_size" in kwargs else 100

        if gensim.__version__[0] == "4":
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
        embeddings = [model_d2v.docvecs[i] for i, e in enumerate(embeddings)]

        return embeddings

    elif method == "lda":
        if not isinstance(corpus[0], list):
            corpus = [c.split() for c in corpus]

        dictionary = corpora.Dictionary(corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in corpus]

        model_lda = LdaMulticore(corpus=bow_corpus, id2word=dictionary, **kwargs)
        embeddings = model_lda[bow_corpus]

        return embeddings

    elif method == "tfidf":
        tfidfvectoriser = TfidfVectorizer(**kwargs)
        tfidfvectoriser.fit(corpus)
        embeddings = tfidfvectoriser.transform(corpus)

        return embeddings
    
    elif method == "wikilinks":
        if os.path.isfile("./wikilinks_embedding_model.h5"):
            model = load_model("./wikilinks_embedding_model.h5")
            layer = model.get_layer('book_embedding')
            weights = layer.get_weights()[0]
            embeddings = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
            return embeddings
        else:
            embeddings = _wikilinks_nn()
            return embeddings


def gen_sim_matrix(
    method="lda", metric="cosine", embeddings=None,
):
    """
    Derives a similarity matrix from document embeddings.

    Parameters
    ----------
        method : str (default=lda)
            The modelling method

            Options:
                BERT: Bidirectional Encoder Representations from Transformers

                    - Words embeddings are derived via Google Neural Networks

                    - Embeddings are then used to derive similarities

                Doc2vec : Document to Vector

                    - An entire document is converted to a vector

                    - Based on word2vec, but maintains the document context

                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories

                    - These categories are then used to classify individual entries given the percent they fall into categories

                TFIDF: Term Frequency Inverse Document Frequency

                    - Word importance increases proportionally to the number of times a word appears in the document while being offset by the number of documents in the corpus that contain the word

                    - These importances are then vectorized and used to relate documents

        metric : str (default=cosine)
            The metric to be used when comparing vectorized corpus entries

            Options include: cosine and euclidean

    Returns
    -------
        sim_matrix : gensim.interfaces.TransformedCorpus or numpy.ndarray
            The similarity sim_matrix for the corpus from the given model
    """
    method = method.lower()

    valid_methods = ["bert", "doc2vec", "lda", "tfidf"]

    if method not in valid_methods:
        raise ValueError(
            "The value for the 'method' argument is invalid. Please choose one of ".join(
                valid_methods
            )
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

    elif method == "tfidf":
        if metric == "cosine":
            sim_matrix = np.dot(  # pylint: disable=no-member
                embeddings, embeddings.T
            ).toarray()

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
            The name of an item or items of interest

        ratings : list (default=None)
            A list of ratings that correspond to each input

            Note: len(ratings) must equal len(inputs)

        titles : lists (default=None)
            The titles of the articles

        sim_matrix : gensim.interfaces.TransformedCorpus or np.ndarray (default=None)
            The similarity sim_matrix for the corpus from the given model

        n : int (default=10)
            The number of items to recommend

        metric : str (default=cosine)
            The metric to be used when comparing vectorized corpus entries

            Options include: cosine and euclidean

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

                else:
                    if ratings:
                        sims = [
                            np.mean([r * s, weights[r] * sim_matrix[i][j]])
                            for j, s in enumerate(sims)
                        ]
                    else:
                        sims = [
                            np.mean([r * s, sim_matrix[i][j]])
                            for j, s in enumerate(sims)
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
        # Cosine similarities have been used (higher is better)
        recommendations = sorted(titles_and_scores, key=lambda x: x[1], reverse=True)
    elif metric == "euclidean":
        # Euclidean distances have been used (lower is better)
        recommendations = sorted(titles_and_scores, key=lambda x: x[1], reverse=False)

    recommendations = [r for r in recommendations if r[0] not in inputs][:n]

    return recommendations

def _wikilinks_nn(path_to_json = None, embedding_size = 50):
    """
    Generates embeddings of wikilinks and articles by training a neural network. Currently only trained on books.  
   
    Parameters
    ----------
        path_to_json : str (default=None)
            The path to the parsed json file. 
        
        embedding_size : int (default = 50)
            The length of the embedding vectors between the articles and the links.
            
    Returns
    -------
        book_weights : np.array
            The normalized embedding vectors for each of the articles. 
            
            Shape of book_weights is (len(books), embedding_size)
    
    """
    if os.path.isfile(path_to_json):
        with open(path_to_json, "r") as fin:
            books = [json.loads(l) for l in fin]
    else:
        raise Exception("Need to parse json for books.")
        
    # Find set of wikilinks for each book and convert to a flattened list
    unique_wikilinks = list(chain(*[list(set(book[2])) for book in books]))
    wikilinks = [link.lower() for link in unique_wikilinks]
    to_remove = ['hardcover', 'paperback', 'hardback', 'e-book', 'wikipedia:wikiproject books', 'wikipedia:wikiproject novels']
    wikilinks = [item for item in wikilinks if item not in to_remove]

    # Limit to wikilinks that occur more than 4 times
    wikilinks_counts = Counter(wikilinks)
    wikilinks_counts = sorted(wikilinks_counts.items(), key = lambda x: x[1], reverse = True)
    wikilinks_counts = OrderedDict(wikilinks_counts)
    links = [t[0] for t in wikilinks_counts.items() if t[1] >= 4]
    
    # map books to their indices, and map links to indices as well 
    book_index = {book[0]: idx for idx, book in enumerate(books)}
    link_index = {link: idx for idx, link in enumerate(links)} 
    
    #Create data from pairs of (book, wikilink) for training the neural network embedding
    pairs = []
    for book in books:
        title = book[0]
        book_links = book[2]
        # Iterate through wikilinks in book article
        for link in book_links:
            # Add index of book and index of link to pairs
            if link.lower() in links:
                pairs.append((book_index[title], link_index[link.lower()]))
    pairs_set = set(pairs)
    
    # Neural network architecture
    # Both inputs are 1-dimensional
    book_input = Input(name = 'book', shape = [1])
    link_input = Input(name = 'link', shape = [1])
    
    # Embedding the book (shape will be (None, 1, 50))
    book_embedding = Embedding(name = 'book_embedding',
                               input_dim = len(book_index),
                               output_dim = embedding_size)(book_input)
    
    # Embedding the link (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'link_embedding',
                               input_dim = len(link_index),
                               output_dim = embedding_size)(link_input)
    
    # Merge the layers with a dot product along the second axis 
    # (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, 
                 axes = 2)([book_embedding, link_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    model = Model(inputs = [book, link], outputs = merged)
    model.compile(optimizer = 'Adam', loss = 'mse')
    
    # Function that creates a generator for training data 
    def _generate_batch(pairs, n_positive = 50, negative_ratio = 1.0):
        """Generate batches of samples for training. 
           Random select positive samples
           from pairs and randomly select negatives."""

        # Create empty array to hold batch
        batch_size = n_positive * (1 + negative_ratio)
        batch = np.zeros((batch_size, 3))

        # Continue to yield samples
        while True:
            # Randomly choose positive examples
            for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
                batch[idx, :] = (book_id, link_id, 1)
            idx += 1

            # Add negative examples until reach batch size
            while idx < batch_size:

                # Random selection
                random_book = random.randrange(len(book_index))
                random_link = random.randrange(len(link_index))

                # Check to make sure this is not a positive example
                if (random_book, random_link) not in pairs_set:

                    # Add to batch and increment index
                    batch[idx, :] = (random_book, random_link, 0)
                    idx += 1

            # Make sure to shuffle order
            np.random.shuffle(batch)
            yield {'book': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]

    n_positive = 1024
    gen = _generate_batch(pairs, n_positive, negative_ratio = 2)
    h = model.fit_generator(gen, epochs = 15, steps_per_epoch = len(pairs) // n_positive)
    
    # Save the model and extract embeddings 
    model.save('./wikilinks_embedding_model.h5')
    
    # Extract embeddings
    book_layer = model.get_layer('book_embedding')
    book_weights = book_layer.get_weights()[0]
    
    # Normalize the weights to have norm of 1 
    book_weights = book_weights / np.linalg.norm(book_weights, axis = 1).reshape((-1, 1))
    
    return book_weights
