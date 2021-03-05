<div align="center">
  <a href="https://github.com/andrewtavis/wikirec"><img src="https://github.com/andrewtavis/wikirec/blob/main/resources/wikirec_logo_transparent.png" width="529" height="169"></a>
</div>

--------------------------------------

[![rtd](https://img.shields.io/readthedocs/wikirec.svg?logo=read-the-docs)](http://wikirec.readthedocs.io/en/latest/)
[![ci](https://img.shields.io/github/workflow/status/andrewtavis/wikirec/CI?logo=github)](https://github.com/andrewtavis/wikirec/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrewtavis/wikirec/branch/main/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/wikirec)
[![pyversions](https://img.shields.io/pypi/pyversions/wikirec.svg?logo=python&logoColor=FFD43B&color=306998)](https://pypi.org/project/wikirec/)
[![pypi](https://img.shields.io/pypi/v/wikirec.svg?color=4B8BBE)](https://pypi.org/project/wikirec/)
[![pypistatus](https://img.shields.io/pypi/status/wikirec.svg)](https://pypi.org/project/wikirec/)
[![license](https://img.shields.io/github/license/andrewtavis/wikirec.svg)](https://github.com/andrewtavis/wikirec/blob/main/LICENSE)
[![contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](https://github.com/andrewtavis/wikirec/blob/main/.github/CONTRIBUTING.md)
[![coc](https://img.shields.io/badge/coc-contributor%20convent-ff69b4.svg)](https://github.com/andrewtavis/wikirec/blob/main/.github/CODE_OF_CONDUCT.md)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Multilingual recommendation engines based on Wikipedia data

[//]: # "The '-' after the section links is needed to make them work on GH (because of ↩s)"
**Jump to:**<a id="jumpto"></a> [Data](#data-) • [Methods](#methods-) • [Recommendations](#recommendations-) • [To-Do](#to-do-)

**wikirec** is a framework that allows users to parse the Wikipedia of any language for entries of a given type and then seamlessly create recommendation engines based on unsupervised natural language processing. The gaol is for wikirec to both refine and deploy models that provide accurate content recommendations based solely on open-source data.

# Installation via PyPi

wikirec can be downloaded from pypi via pip or sourced directly from this repository:

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

# Data [`↩`](#jumpto)

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) allows a user to download Wikipedia texts of a given document type including movies, TV shows, books, music, and countless other classes of information. These texts then serve as the basis to recommend similar content given an input of what the user is interested in.

Article classes are derived from infobox types found on Wikipedia articles. The [article on infoboxes](https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes) (and its translations) contains all the allowed arguments to subset the data by. Simply passing `"Infobox chosen_type"` to the `topic` argument of `data_utils.parse_to_ndjson` in the following example will subset all Wikipedia articles for the given type. For the English Wikipedia, wikirec also provides concise arguments for data that commonly serve as recommendation inputs including: `books`, `songs`, `albums`, `movies`, `tv_series`, `video_games`, as well as various categories of `people` such as `athletes`, `musicians` and `authors`.

Downloading and parsing Wikipedia for the needed data is as simple as:

```python
from wikirec import data_utils

# Downloads the most recent stable bz2 compressed English Wikipedia dump
files = data_utils.download_wiki(language="en")

# Produces an ndjson of all book articles on Wikipedia
data_utils.parse_to_ndjson(
    topic="books",
    output_path="enwiki_books.ndjson",
    multicore=True,
    verbose=True,
)
```

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) also provides a standardized cleaning process for the loaded articles. See [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/wikirec/languages.py) for the full breakdown of what is available for each language.

Generating a clean text and token corpus is achieved through the following:

```python
with open("enwiki_books.ndjson", "r") as fin:
    books = [json.loads(l) for l in fin]

titles = [b[0] for b in books]
texts = [b[1] for b in books]

text_corpus, token_corpus = data_utils.clean(texts=texts)[:2]
```

# Methods [`↩`](#jumpto)

Implemented NLP modeling methods include:

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on NLP models ran over open source Wikipedia data. These representations are leveraged to derive article similarities that are then used to deliver recommendations.

```python
from wikirec import model

# We can pass kwargs for sentence_transformers.SentenceTransformer.encode
sim_matrix = model.gen_sim_matrix(
    method="bert", metric="cosine", corpus=text_corpus,
)
```

### Doc2vec

Doc2vec (a generalization of [Word2vec](https://en.wikipedia.org/wiki/Word2vec)) is an NLP algorithm for deriving vector representations of documents from contextual word interrelations. These representations are then used as a baseline for recommendations.

```python
from wikirec import model

# We can pass kwargs for gensim.models.doc2vec.Doc2Vec
sim_matrix = model.gen_sim_matrix(
    method="doc2vec", metric="cosine", corpus=text_corpus,
)
```

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of wikirec, Wikipedia articles are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics. These topic-word relations are then used to determine article similarities and then make recommendations.

```python
from wikirec import model

# We can pass kwargs for gensim.models.ldamulticore.LdaMulticore
sim_matrix = model.gen_sim_matrix(
    method="lda", metric="cosine", corpus=token_corpus, num_topics=10,
)
```

### TFIDF

[Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. In case of wikirec, word importances are combined and compared to derive article similarities and thus provide recommendations.

```python
from wikirec import model

sim_matrix = model.gen_sim_matrix(
    method="tfidf", metric="cosine", corpus=text_corpus,
)
```

# Recommendations [`↩`](#jumpto)

Once any of the above methods has been trained, generating recommendations is as simple as the following:

```python
from wikirec import model

# Using sim_matrix generated by BERT
recs = model.recommend(
    inputs="title_or_list_of_titles",
    titles=titles,
    sim_matrix=sim_matrix,
    n=10,
)
```

<!---
Outputs
--->

See [examples/rec_books](https://github.com/andrewtavis/wikirec/tree/main/examples/rec_books) and [examples/rec_movies](https://github.com/andrewtavis/wikirec/tree/main/examples/rec_movies) for fully detailed examples with model comparisons.

# To-Do [`↩`](#jumpto)

- Adding further methods for recommendations
- Compiling other sources of open source data that can be used to augment input data
  - Potentially writing scripts to load this data for significant topics
- Allowing multiple infobox topics to be subsetted for at once in [wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) functions
- Updates to [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/wikirec/languages.py) as lemmatization and other linguistic package dependencies evolve
- Creating, improving and sharing [examples](https://github.com/andrewtavis/wikirec/tree/main/examples)
- Updating and refining the [documentation](https://wikirec.readthedocs.io/en/latest/)
- Improving [tests](https://github.com/andrewtavis/wikirec/tree/main/tests) for greater [code coverage](https://codecov.io/gh/andrewtavis/wikirec)

# References
<details><summary><strong>List of references<strong></summary>
<p>

- https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9

- https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

</p>
</details>

# Powered By

<div align="center">
  <a href="https://www.wikipedia.org/"><img height="150" src="https://raw.githubusercontent.com/andrewtavis/wikirec/master/resources/gh_images/wikipedia_logo.png" alt="wikipedia"></a>
</div>
