"""
utils
-----

Utility functions for data loading and cleaning

Contents:
    _combine_tokens_to_str,
    _clean_text_strings,
    clean_and_tokenize_texts,
    prepare_data,
    _prepare_corpus_path
"""

import os
import re
import string
import random
from collections import defaultdict

import spacy
from stopwordsiso import stopwords

from gensim.models import Phrases


def _combine_tokens_to_str(texts, ignore_words=None):
    """
    Combines the texts into one string

    Parameters
    ----------
        texts : str or list
            The texts to be combined

        ignore_words : str or list
            Strings that should be removed from the text body

    Returns
    -------
        texts_str : str
            A string of the full text with unwanted words removed
    """
    if type(texts[0]) == list:
        flat_words = [word for sublist in texts for word in sublist]
    else:
        flat_words = texts

    if type(ignore_words) == str:
        ignore_words = [ignore_words]
    elif ignore_words == None:
        ignore_words = []

    flat_words = [word for word in flat_words if word not in ignore_words]
    texts_str = " ".join([word for word in flat_words])

    return texts_str


def _clean_text_strings(s):
    """
    Cleans the string of a text body to prepare it for BERT analysis

    Parameters
    ----------
        s : str
            The combined texts to be cleaned

    Returns
    -------
        s : str
            The texts formatted for analysis
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1\. \2", s)
    s = s.lower()
    s = re.sub(r"&gt|&lt", " ", s)
    s = re.sub(r"([a-z])\1{2,}", r"\1", s)
    s = re.sub(r"([\W+])\1{1,}", r"\1", s)
    s = re.sub(r"\*|\W\*|\*\W", ". ", s)
    s = re.sub(r"\(.*?\)", ". ", s)
    s = re.sub(r"\W+?\.", ".", s)
    s = re.sub(r"(\.|\?|!)(\w)", r"\1 \2", s)
    s = re.sub(r" ing ", " ", s)
    s = re.sub(r"product received for free[.| ]", " ", s)
    s = re.sub(r"(.{2,}?)\1{1,}", r"\1", s)

    return s.strip()


def lemmatize(tokens, nlp=None):
    """
    Lemmatizes tokens

    Parameters
    ----------
        tokens : list or list of lists

        nlp : spacy.load object
            A spacy language model

    Returns
    -------
        lemmatized_tokens : list or list of lists
            Tokens that have been lemmatized for nlp analysis
    """
    allowed_pos_tags = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]

    lemmatized_tokens = []
    for t in tokens:
        combined_tokens = _combine_tokens_to_str(texts=t)

        lem_tokens = nlp(combined_tokens)
        lemmed_tokens = [
            token.lemma_ for token in lem_tokens if token.pos_ in allowed_pos_tags
        ]

        lemmatized_tokens.append(lemmed_tokens)

    return lemmatized_tokens


def clean_and_tokenize_texts(
    texts, input_language=None, min_freq=2, min_word_len=3, sample_size=1
):
    """
    Cleans and tokenizes a text body to prepare it for analysis

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus, clean_texts, selection_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries
    """
    input_language = input_language.lower()

    if type(texts) == str:
        texts = [texts]

    # Remove spaces that are greater that one in length
    texts_no_large_spaces = []
    for r in texts:
        for i in range(
            25, 0, -1
        ):  # loop backwards to assure that smaller spaces aren't made
            large_space = str(i * " ")
            if large_space in r:
                r = r.replace(large_space, " ")

        texts_no_large_spaces.append(r)

    texts_no_random_punctuation = []
    # Prevent words from being combined when a user types word/word or word-word
    for r in texts_no_large_spaces:
        r = r.replace("/", " ")
        r = r.replace("-", " ")
        if input_language == "fr":
            # Get rid of the 'of' abbreviation for French
            r = r.replace("d'", "")

        texts_no_random_punctuation.append(r)

    # Remove punctuation
    texts_no_punctuation = []
    for r in texts_no_random_punctuation:
        texts_no_punctuation.append(
            r.translate(str.maketrans("", "", string.punctuation + "–" + "’"))
        )

    # Remove stopwords and tokenize
    if stopwords(input_language) != set():  # the input language has stopwords
        stop_words = stopwords(input_language)
    # Stemming and normal stopwords are still full language names
    else:
        stop_words = []

    tokenized_texts = [
        [
            word
            for word in text.lower().split()
            if word not in stop_words and not word.isnumeric()
        ]
        for text in texts_no_punctuation
    ]
    tokenized_texts = [t for t in tokenized_texts if t != []]

    # Add bigrams (first_second word combinations that appear often together)
    tokens_with_bigrams = []
    bigrams = Phrases(
        sentences=tokenized_texts, min_count=3, threshold=5.0
    )  # minimum count for a bigram to be included is 3
    for i, t in enumerate(tokenized_texts):
        for token in bigrams[t]:
            if "_" in token:
                # Token is a bigram, so add it to the tokens
                t.insert(0, token)

        tokens_with_bigrams.append(t)

    # Lemmatize or stem words (try the former first, then the latter)
    nlp = None
    try:
        nlp = spacy.load(input_language)
        lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)

    except OSError:
        try:
            os.system("python -m spacy download {}".format(input_language))
            nlp = spacy.load(input_language)
            lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)
        except:
            pass

    # Remove words that don't appear enough or are too small
    token_frequencies = defaultdict(int)
    for tokens in lemmatized_tokens:
        for t in list(set(tokens)):
            token_frequencies[t] += 1

    if min_word_len == None or min_word_len == False:
        min_word_len = 0
    if min_freq == None or min_freq == False:
        min_freq = 0

    min_len_freq_tokens = []
    for tokens in lemmatized_tokens:
        min_len_freq_tokens.append(
            [
                t
                for t in tokens
                if len(t) >= min_word_len and token_frequencies[t] >= min_freq
            ]
        )

    # Derive those texts that still have valid words
    non_empty_token_indexes = [i for i, t in enumerate(min_len_freq_tokens) if t != []]
    text_corpus = [min_len_freq_tokens[i] for i in non_empty_token_indexes]
    clean_texts = [_clean_text_strings(s=texts[i]) for i in non_empty_token_indexes]

    # Sample words, if necessary
    if sample_size != 1:
        selected_idxs = [
            i
            for i in random.choices(
                range(len(text_corpus)), k=int(sample_size * len(text_corpus))
            )
        ]
    else:
        selected_idxs = list(range(len(text_corpus)))

    text_corpus = [text_corpus[i] for i in selected_idxs]
    clean_texts = [clean_texts[i] for i in selected_idxs]

    return text_corpus, clean_texts, selected_idxs


def prepare_data(
    data=None,
    target_cols=None,
    input_language=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
):
    """
    Prepares input data for analysis

    Parameters
    ----------
        data : pd.DataFrame or csv/xlsx path
            The data in df or path form

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus, clean_texts, selected_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries
    """
    input_language = input_language.lower()

    if type(target_cols) == str:
        target_cols = [target_cols]

    df_texts = data

    # Select columns from which texts should come
    raw_texts = []

    for i in df_texts.index:
        text = ""
        for c in target_cols:
            if type(df_texts.loc[i, c]) == str:
                text += " " + df_texts.loc[i, c]

        text = text[1:]  # remove first blank space
        raw_texts.append(text)

    text_corpus, clean_texts, selected_idxs = clean_and_tokenize_texts(
        texts=raw_texts,
        input_language=input_language,
        min_freq=min_freq,
        min_word_len=min_word_len,
        sample_size=sample_size,
    )

    return text_corpus, clean_texts, selected_idxs


def _prepare_corpus_path(
    text_corpus=None,
    clean_texts=None,
    target_cols=None,
    input_language=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
):
    """
    Checks a text corpus to see if it's a path, and prepares the data if so

    Parameters
    ----------
        text_corpus : str or list or list of lists
            A path or text corpus over which analysis should be done

        clean_texts : str
            The texts formatted for analysis as strings

        target_cols : str or list (default=None)
            The columns in the csv/xlsx or dataframe that contain the text data to be modeled

        input_language : str (default=None)
            The English name of the language in which the texts are found

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus : list or list of lists
            A prepared text corpus for the data in the given path
    """
    if type(text_corpus) == str:
        try:
            os.path.exists(text_corpus)  # a path has been provided
            text_corpus, clean_texts = prepare_data(
                data=text_corpus,
                target_cols=target_cols,
                input_language=input_language,
                min_freq=min_freq,
                min_word_len=min_word_len,
                sample_size=sample_size,
            )[:2]

            return text_corpus, clean_texts

        except:
            pass

    if clean_texts != None:
        return text_corpus, clean_texts

    else:
        return (
            text_corpus,
            [
                _clean_text_strings(_combine_tokens_to_str(texts=t_c))
                for t_c in text_corpus
            ],
        )
