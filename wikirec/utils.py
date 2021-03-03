"""
utils
-----

Utility functions for data loading and cleaning

Contents:
    _combine_tokens_to_str,
    _clean_text_strings,
    clean_and_tokenize_texts,
    _check_str_similarity,
    _check_str_args,
    graph_topic_num_evals,
    english_names_list
"""

import os
import re
from difflib import SequenceMatcher
import string
import random
from collections import defaultdict
import warnings

import numpy as np
from tqdm.auto import tqdm

import spacy
from stopwordsiso import stopwords

import matplotlib.pyplot as plt
import seaborn as sns

from gensim.models import Phrases

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from sentence_transformers import SentenceTransformer

from wikirec import model, topic_model


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
    texts,
    min_freq=2,
    min_word_len=3,
    max_text_len=None,
    remove_names=False,
    sample_size=1,
):
    """
    Cleans and tokenizes a text body to prepare it for analysis

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        max_text_len : int (default=None)
            The maximum allowable length of a text

        remove_names : bool (default=False)
            Whether to remove the most common English names

        sample_size : float (default=1)
            The amount of data to be randomly sampled

    Returns
    -------
        text_corpus, clean_texts, selection_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as tokens and strings, as well as the indexes for selected entries
    """
    if type(texts) == str:
        texts = [texts]

    pbar = tqdm(desc="Cleaning steps complete", total=7, unit="steps")
    # Remove spaces that are greater that one in length
    texts_no_large_spaces = []
    for t in texts:
        for i in range(
            25, 0, -1
        ):  # loop backwards to assure that smaller spaces aren't made
            large_space = str(i * " ")
            if large_space in t:
                t = t.replace(large_space, " ")

        texts_no_large_spaces.append(t)

    texts_no_websites = []
    for t in texts_no_large_spaces:
        websites = [word for word in t if word[:4] == "http"]

        for w in websites:
            t = t.replace(w, "")

        texts_no_websites.append(t)
    pbar.update()

    texts_no_random_punctuation = []
    # Prevent words from being combined when a user types word/word or word-word
    for r in texts_no_large_spaces:
        r = r.replace("/", " ")
        r = r.replace("-", " ")
        r = r.replace(":", " ")  # split categories
        r = re.sub("==[^>]+==", "", r)  # remove headers
        r = re.sub("< !--[^>]+-- >", "", r)  # remove comments

        texts_no_random_punctuation.append(r)

    texts_no_punctuation = []
    for r in texts_no_random_punctuation:
        texts_no_punctuation.append(
            r.translate(str.maketrans("", "", string.punctuation + "–" + "’"))
        )
    pbar.update()

    stop_words = stopwords("en")
    tokenized_texts = [
        [
            word
            for word in text.lower().split()
            if word not in stop_words and not word.isnumeric()
        ]
        for text in texts_no_punctuation
    ]
    tokenized_texts = [t for t in tokenized_texts if t != []]
    pbar.update()

    # Add bigrams (first_second word combinations that appear often together)
    tokens_with_bigrams = []
    bigrams = Phrases(
        sentences=tokenized_texts, min_count=3, threshold=5.0
    )  # minimum count for a bigram to be included is 3, and half the normal threshold
    for i, t in enumerate(tokenized_texts):
        for token in bigrams[t]:
            if "_" in token:
                # Token is a bigram, so add it to the tokens
                t.insert(0, token)

        tokens_with_bigrams.append(t)

    # Remove names after bigrams have been created
    if remove_names:
        tokens_with_bigrams = [
            [t for t in text if t not in [n.lower() for n in english_names_list()]]
            for text in tokens_with_bigrams
        ]
    pbar.update()

    try:
        nlp = spacy.load("en_core_web_sm")
        lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)

    except OSError:
        try:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            lemmatized_tokens = lemmatize(tokens=tokens_with_bigrams, nlp=nlp)
        except:
            pass
    pbar.update()

    token_frequencies = defaultdict(int)
    for tokens in lemmatized_tokens:
        for t in list(set(tokens)):
            token_frequencies[t] += 1

    if min_word_len == None or min_word_len == False:
        min_word_len = 0
    if min_freq == None or min_freq == False:
        min_freq = 0

    assert (
        type(min_word_len) == int
    ), "The 'min_word_len' argument must be an integer if used"
    assert type(min_freq) == int, "The 'min_freq' argument must be an integer if used"

    min_len_freq_tokens = []
    for tokens in lemmatized_tokens:
        min_len_freq_tokens.append(
            [
                t
                for t in tokens
                if len(t) >= min_word_len and token_frequencies[t] >= min_freq
            ]
        )
    pbar.update()

    non_empty_texts = [t for t in min_len_freq_tokens if t != []]
    clean_texts = [
        _clean_text_strings(s=_combine_tokens_to_str(t)) for t in non_empty_texts
    ]

    if max_text_len != None and type(max_text_len) == int:
        text_corpus = [t[:max_text_len] for t in non_empty_texts]
    else:
        text_corpus = non_empty_texts

    # Sample texts if desired
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
    pbar.update()

    return text_corpus, clean_texts, selected_idxs


def _check_str_similarity(str_1, str_2):
    """Checks the similarity of two strings"""
    return SequenceMatcher(None, str_1, str_2).ratio()


def _check_str_args(arguments, valid_args):
    """
    Checks whether a str argument is valid, and makes suggestions if not
    """
    if type(arguments) == str:
        if arguments in valid_args:
            return arguments

        else:
            suggestions = []
            for v in valid_args:
                similarity_score = round(
                    _check_str_similarity(str_1=arguments, str_2=v), 2
                )
                arg_and_score = (v, similarity_score)
                suggestions.append(arg_and_score)

            ordered_suggestions = sorted(suggestions, key=lambda x: x[1], reverse=True)

            print(f"'{arguments}' is not a valid argument for the given function.")
            print(f"The closest valid options to '{arguments}' are:")
            for item in ordered_suggestions[:5]:
                print(item)

            raise ValueError("An invalid string has been passed.")

    elif type(arguments) == list:
        # Check arguments, and remove them if they're invalid
        for a in arguments:
            _check_str_args(arguments=a, valid_args=valid_args)

        return arguments


def graph_topic_num_evals(
    method=["lda", "lda_bert"],
    text_corpus=None,
    clean_texts=None,
    num_topic_words=10,
    topic_nums_to_compare=None,
    min_freq=2,
    min_word_len=3,
    sample_size=1,
    metrics=True,
    verbose=True,
):
    """
    Graphs metrics for the given models over the given number of topics

    Parameters
    ----------
        method : str (default=lda_bert)
            The modelling method

            Options:
                LDA: Latent Dirichlet Allocation

                    - Text data is classified into a given number of categories
                    - These categories are then used to classify individual entries given the percent they fall into categories

                BERT: Bidirectional Encoder Representations from Transformers

                    - Words are classified via Google Neural Networks
                    - Word classifications are then used to derive topics

                LDA_BERT: Latent Dirichlet Allocation with BERT embeddigs

                    - The combination of LDA and BERT via an autoencoder

        text_corpus : list or list of lists
            The text corpus over which analysis should be done

        clean_texts : list
            Text strings that are formatted for cluster models

        num_topic_words : int (default=10)
            The number of keywords that should be extracted

        topic_nums_to_compare : list (default=None)
            The number of topics to compare metrics over

            Note: None selects all numbers from 1 to num_topic_words

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the text corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        sample_size : float (default=None: sampling for non-BERT techniques)
            The size of a sample for BERT models

        metrics : str or bool (default=True: all metrics)
            The metrics to include

            Options:
                stability: model stability based on Jaccard similarity

                coherence: how much the words associated with model topics co-occur

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query

    Returns
    -------
        ax : matplotlib axis
            A graph of the given metrics for each of the given models based on each topic number
    """
    assert (
        metrics == "stability" or metrics == "coherence" or metrics == True
    ), "An invalid value has been passed to the 'metrics' argument - please choose from 'stability', 'coherence', or True for both."

    if metrics == True:
        metrics = ["stability", "coherence"]

    if type(method) == str:
        method = [method]

    method = [m.lower() for m in method]

    def jaccard_similarity(topic_1, topic_2):
        """
        Derives the Jaccard similarity of two topics

        Notes
        -----
            Jaccard similarity:
                - A statistic used for comparing the similarity and diversity of sample sets
                - J(A,B) = (A ∩ B)/(A ∪ B)
                - Goal is low Jaccard scores for coverage of the diverse elements
        """
        # Fix for cases where there are not enough texts for clustering models
        if topic_1 == [] and topic_2 != []:
            topic_1 = topic_2
        if topic_1 != [] and topic_2 == []:
            topic_2 = topic_1
        if topic_1 == [] and topic_2 == []:
            topic_1, topic_2 = ["_None"], ["_None"]
        intersection = set(topic_1).intersection(set(topic_2))
        num_intersect = float(len(intersection))

        union = set(topic_1).union(set(topic_2))
        num_union = float(len(union))

        return num_intersect / num_union

    plt.figure()  # begin figure
    metric_vals = []  # add metric values so that figure y-axis can be scaled

    # Initialize the topics numbers that models should be run for
    if topic_nums_to_compare == None:
        topic_nums_to_compare = list(range(num_topic_words + 2))[1:]
    else:
        # If topic numbers are given, then add one more for comparison
        topic_nums_to_compare = topic_nums_to_compare + [topic_nums_to_compare[-1] + 1]

    bert_model = None
    if "bert" in method or "lda_bert" in method:
        # Multilingual BERT model trained on the top 100+ Wikipedias for semantic textual similarity
        bert_model = SentenceTransformer("xlm-r-bert-base-nli-stsb-mean-tokens")

    ideal_topic_num_dict = {}
    for m in method:
        topics_dict = {}
        stability_dict = {}
        coherence_dict = {}

        disable = not verbose
        for t_n in tqdm(topic_nums_to_compare, desc=f"{m}-topics", disable=disable,):
            tm = topic_model.TopicModel(num_topics=t_n, method=m, bert_model=bert_model)
            tm.fit(
                texts=clean_texts, text_corpus=text_corpus, method=m, m_clustering=None
            )

            # Assign topics given the current number t_n
            topics_dict[t_n] = model._order_and_subset_by_coherence(
                model=tm, num_topics=t_n, num_topic_words=num_topic_words
            )[0]

            coherence_dict[t_n] = model.get_coherence(
                model=tm,
                text_corpus=text_corpus,
                num_topics=t_n,
                num_topic_words=num_topic_words,
                measure="c_v",
            )

        if "stability" in metrics:
            for j in range(0, len(topic_nums_to_compare) - 1):
                jaccard_sims = []
                for t1, topic1 in enumerate(  # pylint: disable=unused-variable
                    topics_dict[topic_nums_to_compare[j]]
                ):
                    sims = []
                    for t2, topic2 in enumerate(  # pylint: disable=unused-variable
                        topics_dict[topic_nums_to_compare[j + 1]]
                    ):
                        sims.append(jaccard_similarity(topic1, topic2))

                    jaccard_sims.append(sims)

                stability_dict[topic_nums_to_compare[j]] = np.array(jaccard_sims).mean()

            mean_stabilities = [
                stability_dict[t_n] for t_n in topic_nums_to_compare[:-1]
            ]
            metric_vals += mean_stabilities

            ax = sns.lineplot(
                x=topic_nums_to_compare[:-1],
                y=mean_stabilities,
                label="{}: Average Topic Overlap".format(m.upper()),
            )

        if "coherence" in metrics:
            coherences = [coherence_dict[t_n] for t_n in topic_nums_to_compare[:-1]]
            metric_vals += coherences

            ax = sns.lineplot(
                x=topic_nums_to_compare[:-1],
                y=coherences,
                label="{}: Topic Coherence".format(m.upper()),
            )

        # If both metrics can be calculated, then an optimal number of topics can be derived
        if "stability" in metrics and "coherence" in metrics:
            coh_sta_diffs = [
                coherences[i] - mean_stabilities[i]
                for i in range(len(topic_nums_to_compare))[:-1]
            ]
            coh_sta_max = max(coh_sta_diffs)
            coh_sta_max_idxs = [
                i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max
            ]
            model_ideal_topic_num_index = coh_sta_max_idxs[
                0
            ]  # take lower topic numbers if more than one max
            model_ideal_topic_num = topic_nums_to_compare[model_ideal_topic_num_index]

            plot_model_ideal_topic_num = model_ideal_topic_num
            if plot_model_ideal_topic_num == topic_nums_to_compare[-1] - 1:
                # Prevents the line from not appearing on the plot
                plot_model_ideal_topic_num = plot_model_ideal_topic_num - 0.005
            elif plot_model_ideal_topic_num == topic_nums_to_compare[0]:
                # Prevents the line from not appearing on the plot
                plot_model_ideal_topic_num = plot_model_ideal_topic_num + 0.005

            ax.axvline(
                x=plot_model_ideal_topic_num,
                label="{} Ideal Num Topics: {}".format(
                    m.upper(), model_ideal_topic_num
                ),
                color="black",
            )

            ideal_topic_num_dict[m] = (model_ideal_topic_num, coh_sta_max)

    # Set plot limits
    y_max = max(metric_vals) + (0.10 * max(metric_vals))
    ax.set_ylim([0, y_max])
    ax.set_xlim([topic_nums_to_compare[0], topic_nums_to_compare[-1] - 1])

    ax.axes.set_title("Method Metrics per Number of Topics", fontsize=25)
    ax.set_ylabel("Metric Level", fontsize=20)
    ax.set_xlabel("Number of Topics", fontsize=20)
    plt.legend(fontsize=20, ncol=len(method))

    return ax


def english_names_list():
    """
    A list of the most common English first names
    """
    en_list = [
        "Mary",
        "Patricia",
        "Jennifer",
        "Linda",
        "Elizabeth",
        "Barbara",
        "Susan",
        "Jessica",
        "Sarah",
        "Karen",
        "Nancy",
        "Lisa",
        "Margaret",
        "Betty",
        "Sandra",
        "Ashley",
        "Dorothy",
        "Kimberly",
        "Emily",
        "Donna",
        "Michelle",
        "Carol",
        "Amanda",
        "Melissa",
        "Deborah",
        "Stephanie",
        "Rebecca",
        "Laura",
        "Sharon",
        "Cynthia",
        "Kathleen",
        "Amy",
        "Shirley",
        "Angela",
        "Helen",
        "Anna",
        "Brenda",
        "Pamela",
        "Nicole",
        "Samantha",
        "Katherine",
        "Emma",
        "Ruth",
        "Christine",
        "Catherine",
        "Debra",
        "Rachel",
        "Carolyn",
        "Janet",
        "Virginia",
        "Maria",
        "Heather",
        "Diane",
        "Julie",
        "Joyce",
        "Victoria",
        "Kelly",
        "Christina",
        "Lauren",
        "Joan",
        "Evelyn",
        "Olivia",
        "Judith",
        "Megan",
        "Cheryl",
        "Martha",
        "Andrea",
        "Frances",
        "Hannah",
        "Jacqueline",
        "Ann",
        "Gloria",
        "Jean",
        "Kathryn",
        "Alice",
        "Teresa",
        "Sara",
        "Janice",
        "Doris",
        "Madison",
        "Julia",
        "Grace",
        "Judy",
        "Abigail",
        "Marie",
        "Denise",
        "Beverly",
        "Amber",
        "Theresa",
        "Marilyn",
        "Danielle",
        "Diana",
        "Brittany",
        "Natalie",
        "Sophia",
        "Rose",
        "Isabella",
        "Alexis",
        "Kayla",
        "Charlotte",
        "James",
        "John",
        "Robert",
        "Michael",
        "William",
        "David",
        "Richard",
        "Joseph",
        "Thomas",
        "Charles",
        "Christopher",
        "Daniel",
        "Matthew",
        "Anthony",
        "Donald",
        "Mark",
        "Paul",
        "Steven",
        "Andrew",
        "Kenneth",
        "Joshua",
        "Kevin",
        "Brian",
        "George",
        "Edward",
        "Ronald",
        "Timothy",
        "Jason",
        "Jeffrey",
        "Ryan",
        "Jacob",
        "Gary",
        "Nicholas",
        "Eric",
        "Jonathan",
        "Stephen",
        "Larry",
        "Justin",
        "Scott",
        "Brandon",
        "Benjamin",
        "Samuel",
        "Frank",
        "Gregory",
        "Raymond",
        "Alexander",
        "Patrick",
        "Jack",
        "Dennis",
        "Jerry",
        "Tyler",
        "Aaron",
        "Jose",
        "Henry",
        "Adam",
        "Douglas",
        "Nathan",
        "Peter",
        "Zachary",
        "Kyle",
        "Walter",
        "Harold",
        "Jeremy",
        "Ethan",
        "Carl",
        "Keith",
        "Roger",
        "Gerald",
        "Christian",
        "Terry",
        "Sean",
        "Arthur",
        "Austin",
        "Noah",
        "Lawrence",
        "Jesse",
        "Joe",
        "Bryan",
        "Billy",
        "Jordan",
        "Albert",
        "Dylan",
        "Bruce",
        "Willie",
        "Gabriel",
        "Alan",
        "Juan",
        "Logan",
        "Wayne",
        "Ralph",
        "Roy",
        "Eugene",
        "Randy",
        "Vincent",
        "Russell",
        "Louis",
        "Philip",
        "Bobby",
        "Johnny",
        "Bradley",
    ]

    return en_list
