"""
data
----

Module for downloading and preparing data

Contents:
    input_conversion_dict,
    download_wiki,
    _process_article,
    _iterate_and_parse_file,
    parse_to_ndjson,
    _combine_tokens_to_str,
    _clean_text_strings,
    clean

    WikiXmlHandler Class
        __init__,
        characters,
        startElement,
        endElement
"""

from collections import defaultdict
import gc
from itertools import chain
import json
import os
import random
import re
import requests
import string
import time
import xml.sax
import warnings

import subprocess
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Threadpool

from tqdm.auto import tqdm

import mwparserfromhell
from bs4 import BeautifulSoup

import spacy
from stopwordsiso import stopwords
from gensim.models import Phrases

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import tensorflow as tf

from wikirec import utils


def input_conversion_dict():
    """
    A dictionary of argument conversions for commonly recommended articles
    """
    input_conversion_dict = {
        "books": "Infobox book",
        "authors": "Infobox writer",
        "albums": "Infobox album",
        "musicians": "Infobox musical artist",
        "songs": "Infobox song",
        "movies": "Infobox film",
        "tv_series": "Infobox television",
        "video_games": "Infobox video game",
        "artists": "Infobox artist",
        "athletes": "Infobox sportsperson",
        "politicians": "Infobox officeholder",
        "people": "Infobox person",
    }

    return input_conversion_dict


def download_wiki(target_dir="wikipedia_dump", explicit_dump=False):
    """
    Downloads the most recent stable dump of the English Wikipedia if it is not already in the specified pwd directory

    Parameters
    ----------
        target_dir : str (default=wikipedia_dump)
            The directory in the pwd into which files should be downloaded

        explicit_dump : str (default=False)
            An explicit Wikipedia dump that the user wants to download

            Note: a value of False will select the third from the last (latest stable dump)

    Returns
    -------
        file_info : list of lists
            Information on the downloaded Wikipedia dump files
    """
    if not os.path.exists(target_dir):
        print(f"Making {target_dir} directory")
        os.makedirs(target_dir)

    base_url = "https://dumps.wikimedia.org/enwiki/"
    index = requests.get(base_url).text
    soup_index = BeautifulSoup(index, "html.parser")

    all_dumps = [a["href"] for a in soup_index.find_all("a") if a.has_attr("href")]
    target_dump = all_dumps[-3]
    if explicit_dump != False:
        if explicit_dump in all_dumps:
            target_dump = explicit_dump

    dump_url = base_url + target_dump
    dump_html = requests.get(dump_url).text
    soup_dump = BeautifulSoup(dump_html, "html.parser")

    files = []

    for file in soup_dump.find_all("li", {"class": "file"}):
        text = file.text
        if "pages-articles-multistream" in text:
            files.append((text.split()[0], text.split()[1:]))

    # Don't select the combined dump so we can check the progress
    files_to_download = [file[0] for file in files if ".xml-p" in file[0]]

    file_info = []

    file_present_bools = [
        os.path.exists(target_dir + "/" + f) for f in files_to_download
    ]
    if len(list(set(file_present_bools))) == 1 and file_present_bools[0] == True:
        dl_files = False
    else:
        dl_files = True

    if dl_files == True:
        for f in files_to_download:
            file_path = target_dir + "/" + f
            if not os.path.exists(file_path):
                saved_file_path = tf.keras.utils.get_file(
                    f,
                    origin=dump_url + f,
                    extract=True,
                    archive_format="auto",
                    cache_subdir=target_dir,
                )

                file_size = os.stat(saved_file_path).st_size / 1e6
                total_articles = int(f.split("p")[-1].split(".")[-2]) - int(
                    f.split("p")[-2]
                )

                file_info.append((f, file_size, total_articles))

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


def _process_article(title, text, template="Infobox book"):
    """
    Process a wikipedia article looking for a given infobox template

    Parameters
    ----------
        title : str
            The title of the article

        text : str
            The text to be processed

        template : str (default=Infobox book)
            The target template for the corpus

    Returns
    -------
        article_data : tuple
            The data from the article
    """
    wikicode = mwparserfromhell.parse(text)
    matching_templates = wikicode.filter_templates(matches=template)
    matching_templates = [
        x
        for x in matching_templates
        if x.name.strip_code().strip().lower() == template.lower()
    ]

    if len(matching_templates) >= 1:
        text = wikicode.strip_code().strip()
        title = re.sub(r"\(.*?\)", "", title).strip()

        article_data = (title, text)

        return article_data


def _iterate_and_parse_file(args):
    """
    Creates partitions of desired articles

    Parameters
    ----------
        args : tuple
            The below arguments as a tuple to allow for pool.imap_unordered rather than pool.starmap

        topic : str
            The topics that articles should be subset by.

            Note: this corresponds to the type of infobox from Wikipedia articles

        input_path : str
            The path to the data file

        partitions_dir : str
            The path to where output file should be stored

        limit : int
            An optional limit of the number of topic articles to find

        verbose : bool
            Whether to show a tqdm progress bar for the processes

    Returns
    -------
        A parsed file Wikipedia dump file with articles of the specified topic
    """
    topic, input_path, partitions_dir, limit, verbose = args

    if not os.path.exists(partitions_dir):
        print(f"Making {partitions_dir} directory for the partitions")
        os.makedirs(partitions_dir)

    if topic in input_conversion_dict().keys():
        topic = input_conversion_dict()[topic]

    handler = WikiXmlHandler()
    handler.template = topic
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    file_name = input_path.split("/")[-1].split("-")[-1].split(".")[-2]
    file_name = f"{file_name}.ndjson"
    output_path = partitions_dir + "/" + file_name

    if not os.path.exists(output_path):
        disable = not verbose

        if limit == None:
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
                disable=disable,
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
                total=limit, desc="Articles found", unit="article", disable=disable,
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
    topic="books",
    output_path="topic_articles",
    input_dir="wikipedia_dump",
    partitions_dir="partitions",
    limit=None,
    delete_parsed_files=False,
    multicore=True,
    verbose=True,
):
    """
    Finds all Wikipedia entries for the given topic and convert them to json files

    Parameters
    ----------
        topic : str (default=books)
            The topics that articles should be subset by.

            Note: this corresponds to the type of infobox from Wikipedia articles

        output_path : str (default=topic_articles)
            The name of the final output ndjson file

        input_dir : str (default=wikipedia_dump)
            The path to the directory where the data is stored

        partitions_dir : str (default=partitions)
            The path to the directory where the output should be stored

        limit : int (default=None)
            An optional limit of the number of topic articles per dump file to find

        delete_parsed_files : bool (default=False)
            Whether to delete the separate parsed files after combining them

        multicore : bool (default=True)
            Whether to use multicore processesing

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the processes

    Returns
    -------
        Wikipedia dump files parsed for the given template type and converted to json files
    """
    output_dir = "/".join([i for i in output_path.split("/")[:-1]])
    if not os.path.exists(output_dir):
        print(f"Making {output_dir} directory for the output")
        os.makedirs(output_dir)

    if topic in input_conversion_dict().keys():
        topic = input_conversion_dict()[topic]

    if multicore == True:
        num_cores = os.cpu_count()
    elif multicore == False:
        num_cores = 1
    elif type(multicore) == int:
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
            [topic] * len(target_files),
            target_files,
            [partitions_dir] * len(target_files),
            [limit] * len(target_files),
            [False] * len(target_files),
        )

        disable = not verbose
        if __name__ == "wikirec.data_utils":
            with Pool(processes=num_cores) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_iterate_and_parse_file, parse_inputs),
                    total=len(target_files),
                    desc="Files partitioned",
                    unit="files",
                    disable=disable,
                ):
                    pass

        def read_and_combine_json(file_path):
            """Read in json data from a file_path"""
            data = []

            with open(file_path, "r") as fin:
                for l in fin.readlines():
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
        print(f"File {output_file_name} with articles for the given topic saved")

    else:
        print(
            f"File {output_file_name} with articles for the given topic already exists"
        )

    if delete_parsed_files:
        if os.path.exists(partitions_dir):
            print(f"Deleting {partitions_dir} directory")
            os.system(f"rm -rf {partitions_dir}")

    return


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


def clean(
    texts,
    min_freq=2,
    min_word_len=3,
    max_text_len=None,
    remove_names=False,
    sample_size=1,
    verbose=True,
):
    """
    Cleans and tokenizes a text body to prepare it for analysis

    Parameters
    ----------
        texts : str or list
            The texts to be cleaned and tokenized

        min_freq : int (default=2)
            The minimum allowable frequency of a word inside the corpus

        min_word_len : int (default=3)
            The smallest allowable length of a word

        max_text_len : int (default=None)
            The maximum allowable length of a text

        remove_names : bool (default=False)
            Whether to remove the most common English names

        sample_size : float (default=1)
            The amount of data to be randomly sampled

        verbose : bool (default=True)
            Whether to show a tqdm progress bar for the query

    Returns
    -------
        text_corpus, token_corpus, selection_idxs : list or list of lists, list, list
            The texts formatted for text analysis both as strings as tokens, as well as the indexes for selected entries
    """
    if type(texts) == str:
        texts = [texts]

    disable = not verbose
    pbar = tqdm(desc="Cleaning steps complete", total=7, unit="steps", disable=disable)
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
            [
                t
                for t in text
                if t not in [n.lower() for n in utils.english_names_list()]
            ]
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
    text_corpus = [
        _clean_text_strings(s=_combine_tokens_to_str(t)) for t in non_empty_texts
    ]

    if max_text_len != None and type(max_text_len) == int:
        token_corpus = [t[:max_text_len] for t in non_empty_texts]
    else:
        token_corpus = non_empty_texts

    # Sample texts if desired
    if sample_size != 1:
        selected_idxs = [
            i
            for i in random.choices(
                range(len(token_corpus)), k=int(sample_size * len(token_corpus))
            )
        ]
    else:
        selected_idxs = list(range(len(token_corpus)))

    text_corpus = [text_corpus[i] for i in selected_idxs]
    token_corpus = [token_corpus[i] for i in selected_idxs]
    pbar.update()

    return text_corpus, token_corpus, selected_idxs


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Parse through XML data using SAX"""

    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self.template = "Infobox book"
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._target_articles = []

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ("title", "text"):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = " ".join(self._buffer)

        if name == "page":
            target_article = _process_article(**self._values, template=self.template)
            if target_article:
                self._target_articles.append(target_article)
