<div align="center">
  <a href="https://github.com/andrewtavis/wikirec"><img src="https://raw.githubusercontent.com/andrewtavis/wikirec/main/resources/wikirec_logo_transparent.png" width="529" height="169"></a>
</div>

---

[![rtd](https://img.shields.io/readthedocs/wikirec.svg?logo=read-the-docs)](http://wikirec.readthedocs.io/en/latest/)
[![ci](https://img.shields.io/github/workflow/status/andrewtavis/wikirec/CI?logo=github)](https://github.com/andrewtavis/wikirec/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrewtavis/wikirec/branch/main/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/wikirec)
[![quality](https://img.shields.io/codacy/grade/96812281c0a1488fbf6e1a09281b485f?logo=codacy)](https://app.codacy.com/gh/andrewtavis/wikirec/dashboard)
[![pyversions](https://img.shields.io/pypi/pyversions/wikirec.svg?logo=python&logoColor=FFD43B&color=306998)](https://pypi.org/project/wikirec/)
[![pypi](https://img.shields.io/pypi/v/wikirec.svg?color=4B8BBE)](https://pypi.org/project/wikirec/)
[![pypistatus](https://img.shields.io/pypi/status/wikirec.svg)](https://pypi.org/project/wikirec/)
[![license](https://img.shields.io/github/license/andrewtavis/wikirec.svg)](https://github.com/andrewtavis/wikirec/blob/main/LICENSE.txt)
[![coc](https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg)](https://github.com/andrewtavis/wikirec/blob/main/.github/CODE_OF_CONDUCT.md)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![colab](https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252)](https://colab.research.google.com/github/andrewtavis/wikirec)

### Recommendation engine framework based on Wikipedia data

**wikirec** is a framework that allows users to parse Wikipedia in any language for entries of a given type and then seamlessly generate recommendations for the given content. Recommendations are based on unsupervised natural language processing over article texts, with ratings being leveraged to weigh inputs and indicate preferences. The goal is for wikirec to both refine and deploy models that provide accurate content recommendations with only open-source data.

See the [documentation](https://wikirec.readthedocs.io/en/latest/) for a full outline of the package including models and data preparation.

# **Contents**<a id="contents"></a>

- [Installation](#installation)
- [Data](#data)
- [Methods](#methods)
- [Recommendations](#recommendations)
- [Comparative Results](#comparative-results)
- [To-Do](#to-do)

# Installation [`↩`](#contents) <a id="installation"></a>

wikirec can be downloaded from PyPI via pip or sourced directly from this repository:

```bash
pip install wikirec
```

```bash
git clone https://github.com/andrewtavis/wikirec.git
cd wikirec
python setup.py install
```

```python
import wikirec
```

# Data [`↩`](#contents) <a id="data"></a>

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) allows a user to download Wikipedia articles of a given topic including movies, TV shows, books, music, and countless other classes of information. These texts then serve as the basis to recommend similar content given an input of what the user is interested in.

Article topics are derived from infobox types found on Wikipedia articles. The [article on infoboxes](https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes) (and its translations) contains all the allowed arguments to subset the data by. Simply passing `"Infobox chosen_type"` to the `topics` argument of `data_utils.parse_to_ndjson()` in the following example will subset all Wikipedia articles for the given type. Lists can also be passed if more than one topic is desired. For the English Wikipedia, wikirec also provides concise arguments for data that commonly serve as recommendation inputs including: `books`, `songs`, `albums`, `movies`, `tv_series`, `video_games`, as well as various categories of `people` such as `athletes`, `musicians` and `authors` (see [data_utils.input_conversion_dict()](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py)).

Data processing in wikirec involves the following steps:

<details><summary><strong>Downloading and Parsing Articles</strong></summary>
<p>

Downloading and parsing Wikipedia articles is as simple as:

```python
from wikirec import data_utils

# Downloads the most recent stable bz2 compressed English Wikipedia dump
files = data_utils.download_wiki(language="en", target_dir="./enwiki_dump")

# Produces an ndjson of all book articles on Wikipedia
data_utils.parse_to_ndjson(
    topics="books",  # ["books", "short_stories", "plays"]
    output_path="./enwiki_books.ndjson",
    input_dir="./enwiki_dump",
    limit=None,  # articles per file to find
    multicore=True,
    verbose=True,
)
```

The [examples](https://github.com/andrewtavis/wikirec/tree/main/examples) directory has a compressed copy of `enwiki_books.ndjson` for testing purposes.

<p>
</details>

<details><summary><strong>Cleaning Parsed Articles</strong></summary>
<p>

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) also provides a standardized multilingual cleaning process for the parsed articles. See [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/languages.py) for a full breakdown of what is available for each language. Generating a clean text corpus is achieved through the following:

```python
import json

with open("./enwiki_books.ndjson", "r") as f:
    books = [json.loads(l) for l in f]

titles = [b[0] for b in books]
texts = [b[1] for b in books]
wikilinks = [b[2] for b in books]  # internal wikipedia links for NN method

text_corpus, selected_idxs = data_utils.clean(
    texts=texts,
    language="en",
    min_token_freq=5,  # 0 for Bert
    min_token_len=3,  # 0 for Bert
    min_tokens=50,
    max_token_index=-1,
    remove_stopwords=True,  # False for Bert
    verbose=True,
)

selected_titles = [titles[i] for i in selected_idxs]
```

From here `text_corpus` would be used to derive article similarities that are then used to make recommendations for any title found in `selected_titles`.

<p>
</details>

# Methods [`↩`](#contents) <a id="methods"></a>

Recommendations in wikirec are generated from similarity matrices derived from trained model embeddings. Implemented NLP modeling methods within [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) include:

<details><summary><strong>BERT</strong></summary>
<p>

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on NLP models ran over open source Wikipedia data. These representations are leveraged to derive article similarities that are then used to deliver recommendations.

wikirec uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers) pretrained models. See their GitHub and [documentation](https://www.sbert.net/) for the available models.

```python
from wikirec import model

# Remove n-grams for BERT training
corpus_no_ngrams = [
    " ".join([t for t in text.split(" ") if "_" not in t]) for text in text_corpus
]

# We can pass kwargs for sentence_transformers.SentenceTransformer.encode
bert_embeddings = model.gen_embeddings(
        method="bert",
        corpus=corpus_no_ngrams,
        bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
        show_progress_bar=True,
        batch_size=32,
)
```

<p>
</details>

<details><summary><strong>Doc2vec</strong></summary>
<p>

A generalization of [Word2vec](https://en.wikipedia.org/wiki/Word2vec), Doc2vec is an NLP algorithm for deriving vector representations of documents from contextual word interrelations. These representations are then used as a baseline for recommendations.

```python
from wikirec import model

# We can pass kwargs for gensim.models.doc2vec.Doc2Vec
d2v_embeddings = model.gen_embeddings(
        method="doc2vec",
        corpus=text_corpus,
        vector_size=100,
        epochs=10,
        alpha=0.025,
)
```

<p>
</details>

<details><summary><strong>LDA</strong></summary>
<p>

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of wikirec, Wikipedia articles are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics. These topic-word relations are then used to determine article similarities and then make recommendations.

```python
from wikirec import model

# We can pass kwargs for gensim.models.ldamulticore.LdaMulticore
lda_embeddings = model.gen_embeddings(
        method="lda",
        corpus=text_corpus,  # automatically tokenized for LDA
        num_topics=50,
        passes=10,
        decay=0.5,
)
```

<p>
</details>

<details><summary><strong>TFIDF</strong></summary>
<p>

[Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. In case of wikirec, word importances are combined and compared to derive article similarities and thus provide recommendations.

```python
from wikirec import model

# We can pass kwargs for sklearn.feature_extraction.text.TfidfVectorizer
tfidf_embeddings = model.gen_embeddings(
        method="tfidf",
        corpus=text_corpus,
        max_features=None,
        norm="l2",
)
```

<p>
</details>

<details><summary><strong>Wikilink NN</strong></summary>
<p>

Based on this [Towards Data Science article](https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9), the wikilinks neural network method makes the assumption that content will be similar if they are linked to the same Wikipedia articles. A corpus of internal wikilinks per article is passed, and embeddings based on these internal references are then derived.

```python
from wikirec import model

wikilink_embeddings = model.gen_embeddings(
        method="WikilinkNN",
        path_to_json="./enwiki_books.ndjson",  # json used instead of a corpus
        embedding_size=50,
)
```

The [examples](https://github.com/andrewtavis/wikirec/tree/main/examples) directory has a copy of `books_embedding_model.h5` for testing purposes.

<p>
</details>

# Recommendations [`↩`](#contents) <a id="recommendations"></a>

After embeddings have been generated we can then create matrices that represent article-article `cosine` or `euclidean` similarities. These can then be sorted and selected from, with the recommendation process being as simple as the following:

```python
from wikirec import model

sim_matrix = model.gen_sim_matrix(
    method="chosen_method",
    metric="cosine",  # euclidean
    embeddings=method_embeddings,
)

recs = model.recommend(
    inputs="title_or_list_of_titles",
    ratings=None,  # list of ints/floats between 0 and 10
    titles=selected_titles,
    sim_matrix=sim_matrix,
    metric="cosine",  # euclidean
    n=10,
)
```

# Comparative Results [`↩`](#contents) <a id="comparative-results"></a>

- TFIDF generally outperformed all other NLP methods in terms of providing what the user would expect, with the results being all the more striking considering its runtime is by far the shortest.
- The other strong performing NLP model is BERT, as it does the best job of providing novel but sensible recommendations.
- The wikilink neural network also provides very sensible results, giving wikirec effective modeling options using different methods.
- LDA with the second shortest runtime provides novel recommendations along with what is expected, but recommends things that seem out of place more often than BERT.
- Doc2vec performs very poorly in that most results are nonsense, and it further takes the longest to train.

See [examples/rec_books](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_books.ipynb) and [examples/rec_movies](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_movies.ipynb) for detailed demonstrations with model comparisons, as well as [examples/rec_ratings](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_ratings.ipynb) for how to leverage user ratings. These notebooks can also be opened in [Google Colab](https://colab.research.google.com/github/andrewtavis/wikirec) for direct experimentation.

Samples of TFIDF, BERT and WikilinkNN book recommendations using cosine similarity follow:

<details><summary><strong>Baseline NLP Models</strong></summary>
<p>

Recommendations for single and multiple inputs follow:

```_output
-- TFIDF --

Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Chamber of Secrets', 0.5974588223913879],
 ['Harry Potter and the Deathly Hallows', 0.5803045645372675],
 ['Harry Potter and the Goblet of Fire', 0.5752151957878091],
 ['Harry Potter and the Half-Blood Prince', 0.5673108963392828],
 ['Harry Potter and the Order of the Phoenix', 0.5662440277414937],
 ['The Magical Worlds of Harry Potter', 0.5098747039144682],
 ['Harry Potter and the Methods of Rationality', 0.5016950079654786],
 ['Harry Potter and the Prisoner of Azkaban', 0.4865186451505909],
 ['Fantastic Beasts and Where to Find Them', 0.4801163347125484],
 ['The Casual Vacancy', 0.44319508498475246]]

The Hobbit recommendations:
[['The History of The Hobbit', 0.7744692537347045],
 ['The Annotated Hobbit', 0.6474663216496771],
 ['Mr. Bliss', 0.5774314075304691],
 ['The Lord of the Rings', 0.5626569367072154],
 ['The Road to Middle-Earth', 0.5386365684368313],
 ['The Marvellous Land of Snergs', 0.5165174723722297],
 ['Tolkien: Maker of Middle-earth', 0.5062523572124091],
 ['The Letters of J. R. R. Tolkien', 0.489393850451095],
 ['The Tolkien Reader', 0.4862696945481724],
 ['J. R. R. Tolkien: A Biography', 0.4813258277958349]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The History of The Hobbit', 0.4144937936077629],
 ['Harry Potter and the Chamber of Secrets', 0.34888387038976304],
 ['The Lord of the Rings', 0.3461664662907625],
 ['The Annotated Hobbit', 0.3431651523791515],
 ['Harry Potter and the Deathly Hallows', 0.3336208844683567],
 ['Harry Potter and the Goblet of Fire', 0.3323377108209634],
 ['Harry Potter and the Half-Blood Prince', 0.32972615751499673],
 ['Mr. Bliss', 0.3219122094772891],
 ['Harry Potter and the Order of the Phoenix', 0.3160426316664049],
 ['The Magical Worlds of Harry Potter', 0.30770960167033506]]

 -- BERT --

 Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Prisoner of Azkaban', 0.8625375],
 ['Harry Potter and the Chamber of Secrets', 0.8557441],
 ['Harry Potter and the Half-Blood Prince', 0.8430752],
 ['Harry Potter and the Goblet of Fire', 0.8258302],
 ['The Magical Worlds of Harry Potter', 0.82496],
 ['A Bad Spell in Yurt', 0.82023925],
 ['Harry Potter and the Order of the Phoenix', 0.80546284],
 ['So You Want to Be a Wizard', 0.803981],
 ['The Weirdstone of Brisingamen', 0.8035261],
 ['Harry Potter and the Cursed Child', 0.79987496]]

 The Hobbit recommendations:
[['The Lord of the Rings', 0.8724792],
 ['Beast', 0.8283818],
 ['The Children of Húrin', 0.8261733],
 ['The Foundling and Other Tales of Prydain', 0.82471454],
 ['The Black Cauldron', 0.82060313],
 ['El Deafo', 0.8167627],
 ['The Little Grey Men', 0.8116319],
 ['The Woggle-Bug Book', 0.8109094],
 ['The Amazing Maurice and His Educated Rodents', 0.8089799],
 ['Dark Lord of Derkholm', 0.8068354]]

 Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The Weirdstone of Brisingamen', 0.79162943],
 ['Harry Potter and the Prisoner of Azkaban', 0.7681779],
 ['A Wizard of Earthsea', 0.7566709],
 ["The Magician's Nephew", 0.75540984],
 ["Merlin's Wood", 0.7530513],
 ['Harry Potter and the Half-Blood Prince', 0.7483348],
 ['Charmed Life', 0.74817574],
 ['The Borrowers Avenged', 0.7475477],
 ["The Inquisitor's Tale", 0.74703705],
 ['The Ghost of Thomas Kempe', 0.74537575]]
```

<p>
</details>

<details><summary><strong>Weighted NLP Approach</strong></summary>
<p>

Better results can be achieved by combining TFIDF and BERT:

```python
tfidf_weight = 0.35
bert_weight = 1.0 - tfidf_weight
bert_tfidf_sim_matrix = tfidf_weight * tfidf_sim_matrix + bert_weight * bert_sim_matrix
```

```_output
-- Weighted BERT and TFIDF --

 Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Chamber of Secrets', 0.7653442323224594],
 ['Harry Potter and the Half-Blood Prince', 0.7465576592959889],
 ['Harry Potter and the Goblet of Fire', 0.7381149146065132],
 ['Harry Potter and the Prisoner of Azkaban', 0.7309308611870757],
 ['Harry Potter and the Order of the Phoenix', 0.7217362181392408],
 ['Harry Potter and the Deathly Hallows', 0.7181677376484684],
 ['The Magical Worlds of Harry Potter', 0.7146800943719254],
 ['Harry Potter and the Cursed Child', 0.6725872668915877],
 ['The Ickabog', 0.6218310147923186],
 ['Fantastic Beasts and Where to Find Them', 0.6161251907593163]]

 The Hobbit recommendations:
[['The History of The Hobbit', 0.78046806361336],
 ['The Lord of the Rings', 0.764041360399863],
 ['The Annotated Hobbit', 0.7444487700381719],
 ['The Marvellous Land of Snergs', 0.6904192459951058],
 ['The Children of Húrin', 0.6804096398917605],
 ['The Road to Middle-Earth', 0.6596135627601877],
 ['Mr. Bliss', 0.6543540064849226],
 ['The Silmarillion', 0.640755416461898],
 ['J. R. R. Tolkien: A Biography', 0.6391232063030203],
 ['Tolkien: Maker of Middle-earth', 0.6309609890944725]]

 Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['Harry Potter and the Half-Blood Prince', 0.6018217616032179],
 ['Harry Potter and the Prisoner of Azkaban', 0.5989788027468591],
 ['The Magical Worlds of Harry Potter', 0.5909785871728664],
 ['Harry Potter and the Order of the Phoenix', 0.5889168038270771],
 ['The Lord of the Rings', 0.5881581367207107],
 ['Harry Potter and the Chamber of Secrets', 0.5868542056295735],
 ['Harry Potter and the Deathly Hallows', 0.5805140956814785],
 ['The Weirdstone of Brisingamen', 0.5725139741586933],
 ['The Children of Húrin', 0.5661655486061915],
 ['Harry Potter and the Goblet of Fire', 0.5653645423523244]]
```

<p>
</details>

<details><summary><strong>Adding User Ratings</strong></summary>
<p>

The `ratings` argument of [wikirec.model.recommend](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) allows users to weight recommendations according to their interests:

```python
model.recommend(
    inputs=[
        "Harry Potter and the Philosopher's Stone",
        "The Hobbit",
        "The Hunger Games",
    ],
    ratings=None,  # averaging their similarity scores
    titles=selected_titles,
    sim_matrix=bert_tfidf_sim_matrix,  # weighted BERT and TFIDF embeddings
    n=20,
    metric="cosine",
)
```

```_output
-- Weighted BERT and TFIDF No Ratings --

[['The Lord of the Rings', 0.8129448240195865],
 ['Harry Potter and the Order of the Phoenix', 0.8058152446026797],
 ['Harry Potter and the Half-Blood Prince', 0.7899741862008424],
 ['Harry Potter and the Prisoner of Azkaban', 0.7795265344828326],
 ['Harry Potter and the Deathly Hallows', 0.774902922811441],
 ['The Weirdstone of Brisingamen', 0.7718548190275306],
 ['The Magical Worlds of Harry Potter', 0.7691708768967348],
 ['Harry Potter and the Chamber of Secrets', 0.7669100258159494],
 ['Gregor and the Curse of the Warmbloods', 0.762141807244329],
 ['The Marvellous Land of Snergs', 0.7591230587892558],
 ['Mockingjay', 0.7585438304114571],
 ['Fantastic Beasts and Where to Find Them', 0.757280478510476],
 ['The Children of Húrin', 0.7570409672927969],
 ['The Book of Three', 0.7497114647690369],
 ['Harry Potter and the Goblet of Fire', 0.7414905999564945],
 ['The Bone Season', 0.7401901103966402],
 ['A Wrinkle in Time', 0.7392014390129485],
 ['A Wizard of Earthsea', 0.7337085913181924],
 ['The Casual Vacancy', 0.7306041913659236],
 ['Krindlekrax', 0.7301903731240345]]
```

```python
model.recommend(
    inputs=[
        "Harry Potter and the Philosopher's Stone",
        "The Hobbit",
        "The Hunger Games",
    ],
    ratings=[7, 6, 9],  # similarity scores weighted by ratings
    titles=selected_titles,
    sim_matrix=bert_tfidf_sim_matrix,  # weighted BERT and TFIDF embeddings
    n=20,
    metric="cosine",
)
```

```_output
-- Weighted BERT and TFIDF With Ratings --

[['Mockingjay', 0.5847107027999907],
 ['Harry Potter and the Order of the Phoenix', 0.5846454899012506],
 ['The Lord of the Rings', 0.5758166462534925],
 ['Harry Potter and the Half-Blood Prince', 0.5677581220645922],
 ['Harry Potter and the Deathly Hallows', 0.5591667887082767],
 ['Harry Potter and the Prisoner of Azkaban', 0.5584267832698454],
 ['Catching Fire', 0.5582404750962344],
 ['Gregor and the Curse of the Warmbloods', 0.5527128074677247],
 ['Harry Potter and the Chamber of Secrets', 0.5524299731616052],
 ['The Weirdstone of Brisingamen', 0.5520358627555212],
 ['The Magical Worlds of Harry Potter', 0.5506942177737976],
 ['The Bone Season', 0.547984210564344],
 ['The Book of Three', 0.5459088891490478],
 ['Fantastic Beasts and Where to Find Them', 0.5443195045210549],
 ['The Marvellous Land of Snergs', 0.5398665287849369],
 ['A Wrinkle in Time', 0.5373739646822866],
 ['The Casual Vacancy', 0.5358385211606874],
 ['Harry Potter and the Goblet of Fire', 0.5346379229854734],
 ['The Children of Húrin', 0.5340832788476909],
 ['A Wizard of Earthsea', 0.5297755576425843]]
```

<p>
</details>

<details><summary><strong>Wikilink NN</strong></summary>
<p>

```_output
-- Wikilink Neural Network Ratings --

[]
```

<p>
</details>

# To-Do [`↩`](#contents) <a id="to-do"></a>

Please see the [contribution guidelines](https://github.com/andrewtavis/wikirec/blob/main/.github/CONTRIBUTING.md) if you are interested in contributing to this project. Work that is in progress or could be implemented includes:

- Allowing a user to express disinterest in [wikirec.model.recommend](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) [(see issue)](https://github.com/andrewtavis/wikirec/issues/33)

- Devising methods to best combine recommendations for more than one input [(see issue)](https://github.com/andrewtavis/wikirec/issues/32)

- Adding arguments to [data_utils.input_conversion_dict()](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) based on Wikipedia languages to simplify parsing arguments [(see issue)](https://github.com/andrewtavis/wikirec/issues/34)

- Adding and refining methods for recommendations in [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) [(see issue)](https://github.com/andrewtavis/wikirec/issues/31)

- Adding embeddings visualization methods to wikirec [(see issue)](https://github.com/andrewtavis/wikirec/issues/35)

- Creating, improving and sharing [examples](https://github.com/andrewtavis/wikirec/tree/main/examples)

- Compiling other sources of open-source data that can be used to augment input data

  - Potentially writing scripts to load this data for significant topics

- Updates to [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/languages.py) as lemmatization and other linguistic package dependencies evolve

- Allowing euclidean distance measurements for LDA based recommendations in [wikirec.model.gen_sim_matrix()](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py)

- Expanding the [documentation](https://wikirec.readthedocs.io/en/latest/)

- Improving [tests](https://github.com/andrewtavis/wikirec/tree/main/tests) for greater [code coverage](https://codecov.io/gh/andrewtavis/wikirec)

# References

<details><summary><strong>List of references</strong></summary>
<p>

- https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9

- https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

</p>
</details>

# Powered By

<div align="center">
  <a href="https://www.wikipedia.org/"><img height="150" src="https://raw.githubusercontent.com/andrewtavis/wikirec/master/resources/gh_images/wikipedia_logo.png" alt="wikipedia"></a>
</div>
