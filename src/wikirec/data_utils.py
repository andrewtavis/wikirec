"""
data
----

Module for downloading and preparing data.

Contents:
    input_conversion_dict,
    download_wiki,
    _process_article,
    iterate_and_parse_file,
    parse_to_ndjson,
    _combine_tokens_to_str,
    _lower_remove_unwanted,
    _lemmatize,
    _subset_and_combine_tokens
    clean

    WikiXmlHandler Class
        __init__,
        characters,
        startElement,
        endElement
"""

import gc
import json
import os
import re
import string
import subprocess
import time
import warnings
import xml.sax
from collections import defaultdict
from itertools import chain
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Threadpool

import defusedxml.sax
import mwparserfromhell
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from tqdm.auto import tqdm

try:
    from nltk.corpus import names
except ImportError:
    import nltk

    nltk.download("names")
    from nltk.corpus import names

male_names = names.words("male.txt")
female_names = names.words("female.txt")
all_names = set(list(male_names) + list(female_names))

import gensim
import spacy
from gensim.models import Phrases
from stopwordsiso import stopwords

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import tensorflow as tf

from wikirec import languages


def input_conversion_dict():
    """
    A dictionary of argument conversions for commonly recommended articles.
    """
    return {
        "en": {
            "books": "Infobox book",
            "short_stories": "Infobox short story",
            "plays": "Infobox play",
            "authors": "Infobox writer",
            "albums": "Infobox album",
            "musicians": "Infobox musical artist",
            "songs": "Infobox song",
            "movies": "Infobox film",
            "films": "Infobox film",
            "tv_series": "Infobox television",
            "video_games": "Infobox video game",
            "artists": "Infobox artist",
            "athletes": "Infobox sportsperson",
            "politicians": "Infobox officeholder",
            "people": "Infobox person",
        }
    }


def download_wiki(language="en", target_dir="wiki_dump", file_limit=-1, dump_id=False):
    """
    Downloads the most recent stable dump of the English Wikipedia if it is not already in the specified pwd directory.

    Parameters
    ----------
        language : str (default=en)
            The language of Wikipedia to download.

        target_dir : str (default=wiki_dump)
            The directory in the pwd into which files should be downloaded.

        file_limit : int (default=-1, all files)
            The limit for the number of files to download.

        dump_id : str (default=False)
            The id of an explicit Wikipedia dump that the user wants to download.

            Note: a value of False will select the third from the last (latest stable dump).

    Returns
    -------
        file_info : list of lists
            Information on the downloaded Wikipedia dump files.
    """
    assert isinstance(
        file_limit, int
    ), "The 'file_limit' argument must be an integer to subset the available file list by as an upper bound."

    if not os.path.exists(target_dir):
        print(f"Making {target_dir} directory")
        os.makedirs(target_dir)

    base_url = f"https://dumps.wikimedia.org/{language}wiki/"
    index = requests.get(base_url).text
    soup_index = BeautifulSoup(index, "html.parser")

    all_dumps = [a["href"] for a in soup_index.find_all("a") if a.has_attr("href")]
    target_dump = all_dumps[-3]
    if dump_id != False and dump_id in all_dumps:
        target_dump = dump_id

    dump_url = base_url + target_dump
    dump_html = requests.get(dump_url).text
    soup_dump = BeautifulSoup(dump_html, "html.parser")

    files = []

    for file in soup_dump.find_all("li", {"class": "file"}):
        text = file.text
        if "pages-articles-multistream" in text:
            files.append((text.split()[0], text.split()[1:]))

    # Don't select the combined dump so we can check the progress
    files_to_download = [file[0] for file in files if ".xml-p" in file[0]][:file_limit]

    file_info = []

    file_present_bools = [
        os.path.exists(target_dir + "/" + f) for f in files_to_download
    ]
    if len(list(set(file_present_bools))) == 1 and file_present_bools[0] == True:
        dl_files = False
    else:
        dl_files = True

    cache_subdir = target_dir.split("/")[-1]
    cache_dir = "/".join(target_dir.split("/")[:-1])
    if cache_dir == "":
        cache_subdir = target_dir
        cache_dir = "."

    if dl_files:
        for f in files_to_download:
            file_path = target_dir + "/" + f
            if not os.path.exists(file_path):
                print(f"DL file to {file_path}")
                saved_file_path = tf.keras.utils.get_file(
                    fname=f,
                    origin=dump_url + f,
                    extract=True,
                    archive_format="auto",
                    cache_subdir=cache_subdir,
                    cache_dir=cache_dir,
                )

                file_size = os.stat(saved_file_path).st_size / 1e6
                total_articles = int(f.split("p")[-1].split(".")[-2]) - int(
                    f.split("p")[-2]
                )

                file_info.append((f.split("-")[-1], file_size, total_articles))

    else:
        print(f"Files already available in the {target_dir} directory.")
        for f in files_to_download:
            file_path = target_dir + "/" + f

            file_size = os.stat(file_path).st_size / 1e6
            total_articles = int(f.split("p")[-1].split(".")[-2]) - int(
                f.split("p")[-2]
            )

            file_info.append((f.split("-")[-1], file_size, total_articles))

    return file_info


def _process_article(title, text, templates="Infobox book"):
    """
    Process a wikipedia article looking for given infobox templates.

    Parameters
    ----------
        title : str
            The title of the article.

        text : str
            The text to be processed.

        templates : str (default=Infobox book)
            The target templates for the corpus.

    Returns
    -------
        title, text, wikilinks: string, string, list
            The data from the article.
    """
    wikicode = mwparserfromhell.parse(text)

    if isinstance(templates, str):
        templates = [templates]

    matching_templates = [wikicode.filter_templates(matches=t) for t in templates]
    matching_templates = [
        x
        for x in [temp for sub_temps in matching_templates for temp in sub_temps]
        if x.name.strip_code().strip().lower() in [t.lower() for t in templates]
    ]

    if matching_templates:
        title = title.strip()
        text = wikicode.strip_code().strip()
        wikilinks = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]

        return title, text, wikilinks


def iterate_and_parse_file(args):
    """
    Creates partitions of desired articles.

    Parameters
    ----------
        args : tuple
            The below arguments as a tuple to allow for pool.imap_unordered rather than pool.starmap.

        topics : str
            The topics that articles should be subset by.

            Note: this corresponds to the type of infobox from Wikipedia articles.

        language : str (default=en)
            The language of Wikipedia that articles are being parsed for.

        input_path : str
            The path to the data file.

        partitions_dir : str
            The path to where output file should be stored.

        limit : int optional (default=None)
            An optional limit of the number of articles to find.

        verbose : bool
            Whether to show a tqdm progress bar for the processes.

    Returns
    -------
        A parsed file Wikipedia dump file with articles of the specified topics.
    """
    topics, language, input_path, partitions_dir, limit, verbose = args

    if not os.path.exists(partitions_dir):
        print(f"Making {partitions_dir} directory for the partitions")
        os.makedirs(partitions_dir)

    if isinstance(topics, str):
        topics = [topics]

    for i, t in enumerate(topics):
        if (
            language in input_conversion_dict().keys()
            and t in input_conversion_dict()[language].keys()
        ):
            topics[i] = input_conversion_dict()[language][t]

    handler = WikiXmlHandler()
    handler.templates = topics
    parser = defusedxml.sax.make_parser()
    parser.setContentHandler(handler)

    file_name = input_path.split("/")[-1].split("-")[-1].split(".")[-2]
    file_name = f"{file_name}.ndjson"
    output_path = partitions_dir + "/" + file_name

    if not os.path.exists(output_path):
        if limit is None:
            pbar = tqdm(
                total=len(
                    [
                        i
                        for i, line in enumerate(
                            subprocess.Popen(
                                ["bzcat"],
                                stdin=open(input_path),
                                stdout=subprocess.PIPE,
                            ).stdout
                        )
                    ]
                ),
                desc="Lines read",
                unit="lines",
                disable=not verbose,
            )
            for i, line in enumerate(  # pylint: disable=unused-variable
                subprocess.Popen(
                    ["bzcat"], stdin=open(input_path), stdout=subprocess.PIPE
                ).stdout
            ):
                try:
                    parser.feed(line)
                except StopIteration:
                    break

                pbar.update()

        else:
            articles_found = 0
            pbar = tqdm(
                total=limit, desc="Articles found", unit="article", disable=not verbose,
            )
            for i, line in enumerate(  # pylint: disable=unused-variable
                subprocess.Popen(
                    ["bzcat"], stdin=open(input_path), stdout=subprocess.PIPE
                ).stdout
            ):
                try:
                    parser.feed(line)
                except StopIteration:
                    break

                if len(handler._target_articles) == articles_found + 1:
                    articles_found += 1
                    pbar.update()

                if len(handler._target_articles) >= limit:
                    break

        with open(output_path, "w") as fout:
            for ta in handler._target_articles:
                fout.write(json.dumps(ta) + "\n")

        if verbose:
            print(
                f"File {file_name} with {len(handler._target_articles)} topic articles processed and saved in {partitions_dir}"
            )

    else:
        if verbose:
            print(f"File {file_name} already exists in {partitions_dir}")

    del handler
    del parser
    gc.collect()

    return None


def parse_to_ndjson(
    topics="books",
    language="en",
    output_path="topic_articles",
    input_dir="wikipedia_dump",
    partitions_dir="partitions",
    limit=None,
    delete_parsed_files=False,
    multicore=True,
    verbose=True,
):
    """
    Finds all Wikipedia entries for the given topics and convert them to json files.

    Parameters
    ----------
        topics : str (default=books)
            The topics that articles should be subset by.

            Note: this corresponds to the type of infobox from Wikipedia articles.

        language : str (default=en)
            The language of Wikipedia that articles are being parsed for.

        output_path : str (default=topic_articles)
            The name of the final output ndjson file.

        input_dir : str (default=wikipedia_dump)
            The path to the directory where the data is stored.

        partitions_dir : str (default=partitions)
            The path to the directory where the output should be stored.

        limit : int (default=None)
            An optional limit of the number of topic articles per dump file to find.

        delete_parsed_files : bool (default=False)
            Whether to delete the separate parsed files after combining them.

        multicore : bool (default=True)
            Whether to use multicore processesing.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the processes.

    Returns
    -------
        Wikipedia dump files parsed for the given template types and converted to json files.
    """
    output_dir = "/".join([i for i in output_path.split("/")[:-1]])
    if not os.path.exists(output_dir):
        print(f"Making {output_dir} directory for the output")
        os.makedirs(output_dir)

    if isinstance(topics, str):
        topics = [topics]

    for i, t in enumerate(topics):
        if language in input_conversion_dict().keys():
            if t in input_conversion_dict()[language].keys():
                topics[i] = input_conversion_dict()[language][t]

    if multicore == True:
        num_cores = os.cpu_count()
    elif multicore == False:
        num_cores = 1
    elif isinstance(multicore, int):
        num_cores = multicore

    if output_path == None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_path = "parsed_data" + timestr
        output_file_name = output_path + ".ndjson"

    else:
        if output_path[-len(".ndjson") :] != ".ndjson":
            output_file_name = output_path + ".ndjson"
        else:
            output_file_name = output_path

    if not os.path.exists(output_file_name):
        if not os.path.exists(partitions_dir):
            print(f"Making {partitions_dir} directory for the partitions")
            os.makedirs(partitions_dir)

        target_files = [
            input_dir + "/" + f for f in os.listdir(input_dir) if "pages-articles" in f
        ]

        parse_inputs = zip(
            [topics] * len(target_files),
            [language] * len(target_files),
            target_files,
            [partitions_dir] * len(target_files),
            [limit] * len(target_files),
            [False] * len(target_files),
        )

        if __name__ == "wikirec.data_utils":
            with Pool(processes=num_cores) as pool:
                for _ in tqdm(
                    pool.imap_unordered(iterate_and_parse_file, parse_inputs),
                    total=len(target_files),
                    desc="Files partitioned",
                    unit="file",
                    disable=not verbose,
                ):
                    pass

        def read_and_combine_json(file_path):
            """
            Read in json data from a file_path.
            """
            data = []

            with open(file_path, "r") as f:
                for l in f.readlines():
                    data.append(json.loads(l))

            return data

        threadpool = Threadpool(processes=num_cores)
        partition_files = [
            partitions_dir + "/" + f
            for f in os.listdir(partitions_dir)
            if f[-len(".ndjson") :] == ".ndjson"
        ]

        if __name__ == "wikirec.data_utils":
            results = threadpool.map(read_and_combine_json, partition_files)

        file_list = list(chain(*results))

        with open(output_file_name, "wt") as fout:
            for f in file_list:
                fout.write(json.dumps(f) + "\n")
        print(f"File {output_file_name} with articles for the given topics saved")

    else:
        print(
            f"File {output_file_name} with articles for the given topics already exists"
        )

    if delete_parsed_files:
        if os.path.exists(partitions_dir):
            print(f"Deleting {partitions_dir} directory")
            os.system(f"rm -rf {partitions_dir}")

    return


def _combine_tokens_to_str(tokens):
    """
    Combines the texts into one string.

    Parameters
    ----------
        tokens : str or list
            The texts to be combined.

    Returns
    -------
        texts_str : str
            A string of the full text with unwanted words removed.
    """
    if isinstance(tokens[0], list):
        flat_words = [word for sublist in tokens for word in sublist]
    else:
        flat_words = tokens

    return " ".join(flat_words)


def _lower_remove_unwanted(args):
    """
    Lower cases tokens and removes numbers and possibly names.

    Parameters
    ----------
        args : list of tuples
            The following arguments zipped.

        text : list
            The text to clean.

        remove_names : bool
            Whether to remove names.

        words_to_ignore : str or list
                Strings that should be removed from the text body.

        stop_words : str or list
            Stopwords for the given language.

    Returns
    -------
        text_lower : list
            The text with lowercased tokens and without unwanted tokens.
    """
    text, remove_names, words_to_ignore, stop_words = args

    if remove_names:
        # Remove names, numbers, words_to_ignore and stop_words after n-grams have been created.
        return [
            token.lower()
            for token in text
            if token not in all_names
            and not token.isnumeric()
            and token not in words_to_ignore
            and token != "ref"
            and token not in stop_words
        ]
    else:
        # Or simply lower case tokens and remove non-bigrammed numbers, words_to_ignore and stop_words.
        return [
            token.lower()
            for token in text
            if not token.isnumeric()
            and token not in words_to_ignore
            and token != "ref"
            and token not in stop_words
        ]


def _lemmatize(tokens, nlp=None, verbose=True):
    """
    Lemmatizes tokens.

    Parameters
    ----------
        tokens : list or list of lists
            Tokens to be lemmatized.

        nlp : spacy.load object
            A spacy language model.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        lemmatized_tokens : list or list of lists
            Tokens that have been lemmatized for nlp analysis.
    """
    allowed_pos_tags = ["NOUN", "PROPN", "ADJ", "ADV", "VERB"]

    lemmatized_tokens = []
    for t in tqdm(
        tokens,
        total=len(tokens),
        desc="Texts lemmatized",
        unit="texts",
        disable=not verbose,
    ):
        combined_tokens = _combine_tokens_to_str(tokens=t)

        lem_tokens = nlp(combined_tokens)
        lemmed_tokens = [
            token.lemma_ for token in lem_tokens if token.pos_ in allowed_pos_tags
        ]

        lemmatized_tokens.append(lemmed_tokens)

    return lemmatized_tokens


def _subset_and_combine_tokens(args):
    """
        Subsets a text by a maximum length and combines it to a string.

        Parameters
        ----------
            args : list of tuples
                The following arguments zipped.

            text : list
                The list of tokens to be subsetted for and combined.

            max_token_index : int (default=-1)
                The maximum allowable length of a tokenized text.

        Returns
        -------
            sub_comb_text : tuple
                An index and its combined text.
        """
    text, max_token_index = args

    return [
        text[0],
        _combine_tokens_to_str(tokens=text[1][:max_token_index]),
    ]


def clean(
    texts,
    language="en",
    min_token_freq=2,
    min_token_len=3,
    min_tokens=0,
    max_token_index=-1,
    min_ngram_count=3,
    remove_stopwords=True,
    ignore_words=None,
    remove_names=False,
    sample_size=1,
    verbose=True,
):
    """
    Cleans text body to prepare it for analysis.

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized.

        language : str (default=en)
            The language of Wikipedia to download.

        min_token_freq : int (default=2)
            The minimum allowable frequency of a word inside the corpus.

        min_token_len : int (default=3)
            The smallest allowable length of a word.

        min_tokens : int (default=0)
            The minimum allowable length of a tokenized text.

        max_token_index : int (default=-1)
            The maximum allowable length of a tokenized text.

        min_ngram_count : int (default=5)
            The minimum occurrences for an n-gram to be included.

        remove_stopwords : bool (default=True)
            Whether to remove stopwords.

        ignore_words : str or list
            Strings that should be removed from the text body.

        remove_names : bool (default=False)
            Whether to remove common names.

        sample_size : float (default=1)
            The amount of data to be randomly sampled.

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query.

    Returns
    -------
        text_corpus, selected_idxs : list, list
            The texts formatted for text analysis as well as the indexes for selected entries.
    """
    language = language.lower()

    # Select abbreviation for the lemmatizer, if it's available.
    if language in languages.lem_abbr_dict().keys():
        language = languages.lem_abbr_dict()[language]

    if isinstance(texts, str):
        texts = [texts]

    if isinstance(ignore_words, str):
        words_to_ignore = [ignore_words]
    elif ignore_words is None:
        words_to_ignore = []

    stop_words = []
    if remove_stopwords:
        if stopwords(language) != set():  # the input language has stopwords
            stop_words = stopwords(language)

        # Stemming and normal stopwords are still full language names.
        elif language in languages.stem_abbr_dict().keys():
            stop_words = stopwords(languages.stem_abbr_dict()[language])

        elif language in languages.sw_abbr_dict().keys():
            stop_words = stopwords(languages.sw_abbr_dict()[language])

    pbar = tqdm(
        desc="Cleaning steps complete", total=7, unit="step", disable=not verbose
    )
    # Remove spaces that are greater that one in length.
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
        websites = [word for word in t.split() if word[:4] == "http"]

        for w in websites:
            t = t.replace(w, "")

        texts_no_websites.append(t)

    # Remove the references section but maintain the categories if they exist.
    # The reference are in the text, so this just removes the section and external links.
    # References are maintained for references like awards.
    texts_no_references = []
    for t in texts_no_websites:
        if "Category:" in t:
            t = re.sub(r"(?<= ==References==).+?(?= Category)", "", t, flags=re.DOTALL)
        else:
            t = t.split("==References==")[0]

        texts_no_references.append(t)

    gc.collect()
    pbar.update()

    texts_no_random_punctuation = []
    # Prevent words from being combined when a user types word/word or word-word or word:word.
    for t in texts_no_references:
        t = t.replace("/", " ")
        t = t.replace("-", " ")
        t = t.replace(":", " ")  # split categories so they can be n-grammed
        t = re.sub("==[^>]+==", "", t)  # remove headers
        t = re.sub("< !--[^>]+-- >", "", t)  # remove comments

        texts_no_random_punctuation.append(t)

    texts_no_punctuation = [
        r.translate(str.maketrans("", "", string.punctuation + "–" + "’"))
        for r in texts_no_random_punctuation
    ]

    # We lower case after names are removed to allow for filtering out capitalized words.
    tokenized_texts = [text.split() for text in texts_no_punctuation]

    gc.collect()
    pbar.update()

    # Add bigrams and trigrams.
    # Use half the normal threshold.
    if float(gensim.__version__[0]) >= 4:
        bigrams = Phrases(
            sentences=tokenized_texts,
            min_count=min_ngram_count,
            threshold=5.0,
            connector_words=stop_words,
        )  # half the normal threshold
        trigrams = Phrases(
            sentences=bigrams[tokenized_texts],
            min_count=min_ngram_count,
            threshold=5.0,
            connector_words=stop_words,
        )
    else:
        bigrams = Phrases(  # pylint: disable=unexpected-keyword-arg
            sentences=tokenized_texts,
            min_count=min_ngram_count,
            threshold=5.0,
            common_terms=stop_words,
        )
        trigrams = Phrases(  # pylint: disable=unexpected-keyword-arg
            sentences=bigrams[tokenized_texts],
            min_count=min_ngram_count,
            threshold=5.0,
            common_terms=stop_words,
        )

    tokens_with_ngrams = []
    for text in tqdm(
        tokenized_texts,
        total=len(tokenized_texts),
        desc="n-grams generated",
        unit="texts",
        disable=not verbose,
    ):
        for token in bigrams[text]:
            if token.count("_") == 1:
                # Token is a bigram, so add it to the tokens.
                text.insert(0, token)

        for token in trigrams[bigrams[text]]:
            if token.count("_") == 2:
                # Token is a trigram, so add it to the tokens.
                text.insert(0, token)

        tokens_with_ngrams.append(text)

    gc.collect()
    pbar.update()

    args = zip(
        tokens_with_ngrams,
        [remove_names] * len(tokens_with_ngrams),
        [words_to_ignore] * len(tokens_with_ngrams),
        [stop_words] * len(tokens_with_ngrams),
    )

    num_cores = os.cpu_count()
    if __name__ == "wikirec.data_utils":
        with Pool(processes=num_cores) as pool:
            tokens_lower = list(
                tqdm(
                    pool.imap(_lower_remove_unwanted, args),
                    total=len(tokens_with_ngrams),
                    desc="Unwanted words removed",
                    unit="texts",
                    disable=not verbose,
                )
            )

    gc.collect()
    pbar.update()

    # Try lemmatization, and if not available stem, and if not available nothing.
    try:
        nlp = spacy.load(language)
        base_tokens = _lemmatize(tokens=tokens_lower, nlp=nlp, verbose=verbose)

    except OSError:
        try:
            os.system("python -m spacy download {}".format(language))
            nlp = spacy.load(language)
            base_tokens = _lemmatize(tokens=tokens_lower, nlp=nlp, verbose=verbose)

        except OSError:
            nlp = None

    if nlp is None:
        # Lemmatization failed, so try stemming.
        stemmer = None
        if language in SnowballStemmer.languages:
            stemmer = SnowballStemmer(language)

        # Correct if the abbreviations were put in.
        elif language == "ar":
            stemmer = SnowballStemmer("arabic")

        elif language == "fi":
            stemmer = SnowballStemmer("finish")

        elif language == "hu":
            stemmer = SnowballStemmer("hungarian")

        elif language == "sv":
            stemmer = SnowballStemmer("swedish")

        if stemmer is None:
            # We cannot lemmatize or stem.
            base_tokens = tokens_lower

        else:
            # Stemming instead of lemmatization.
            base_tokens = []
            for tokens in tqdm(
                tokens_lower,
                total=len(tokens_lower),
                desc="Texts stemmed",
                unit="texts",
                disable=not verbose,
            ):
                stemmed_tokens = [stemmer.stem(t) for t in tokens]
                base_tokens.append(stemmed_tokens)

    gc.collect()
    pbar.update()

    token_frequencies = defaultdict(int)
    for tokens in base_tokens:
        for t in list(set(tokens)):
            token_frequencies[t] += 1

    if min_token_len is None or min_token_len == False:
        min_token_len = 0
    if min_token_freq is None or min_token_freq == False:
        min_token_freq = 0

    assert isinstance(
        min_token_len, int
    ), "The 'min_token_len' argument must be an integer if used."
    assert isinstance(
        min_token_freq, int
    ), "The 'min_token_freq' argument must be an integer if used."

    min_len_freq_tokens = [
        [
            t
            for t in tokens
            if len(t) >= min_token_len and token_frequencies[t] >= min_token_freq
        ]
        for tokens in base_tokens
    ]

    gc.collect()
    pbar.update()

    # Save original length for sampling.
    original_len = len(min_len_freq_tokens)
    min_sized_texts = [
        [i, t] for i, t in enumerate(min_len_freq_tokens) if len(t) > min_tokens
    ]

    args = zip(min_sized_texts, [max_token_index] * len(min_sized_texts))
    if __name__ == "wikirec.data_utils":
        with Pool(processes=num_cores) as pool:
            text_corpus = list(
                tqdm(
                    pool.imap(_subset_and_combine_tokens, args),
                    total=len(min_sized_texts),
                    desc="Texts finalized",
                    unit="texts",
                    disable=not verbose,
                )
            )

    gc.collect()

    # Sample texts.
    if len(text_corpus) > int(sample_size * original_len):
        idxs = [t[0] for t in text_corpus]
        selected_idxs = np.random.choice(
            a=idxs, size=int(sample_size * original_len), replace=False
        )

    else:
        selected_idxs = [t[0] for t in text_corpus]

    text_corpus = [t[1] for t in text_corpus if t[0] in selected_idxs]
    pbar.update()

    return text_corpus, selected_idxs


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """
    Parse through XML data using SAX.
    """

    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self.templates = "Infobox book"
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._target_articles = []

    def characters(self, content):
        """
        Characters between opening and closing tags.
        """
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """
        Opening tag of element.
        """
        if name in ("title", "text"):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """
        Closing tag of element.
        """
        if name == self._current_tag:
            self._values[name] = " ".join(self._buffer)

        if name == "page":
            target_article = _process_article(**self._values, templates=self.templates)
            if target_article and (
                "Wikipedia:" not in target_article[0]
                and "Draft:" not in target_article[0]
            ):  # no archive files or drafts
                self._target_articles.append(target_article)
