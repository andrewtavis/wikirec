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

### Open-source recommendation engines based on Wikipedia data

[//]: # "The '-' after the section links is needed to make them work on GH (because of ↩s)"
**Jump to:**<a id="jumpto"></a> [Data](#data-) • [Methods](#methods-) • [Usage](#usage-) • [To-Do](#to-do-)

**wikirec** is a framework that allows users to parse Wikipedia for entries of a given type and then seamlessly create recommendation engines based on unsupervised natural language processing. The gaol is for wikirec to both refine and deploy models that provide accurate content recommendations based solely on open-source data.

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

wikirec allows a user to download Wikipedia texts of a given document type including movies, TV shows, books, music, and countless other classes of information. These texts then serve as the basis to recommend similar content given an input of what the user is interested in.

Article classes are derived from infobox types found on Wikipedia articles. The [article on infoboxes](https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes) contains all the allowed arguments to subset the data by. Simply passing `"Infobox chosen_type"` to the `topic` argument in the following example will subset all Wikipedia articles for the given type. wikirec also provides concise arguments for data that commonly serve as recommendation inputs including: `books`, `songs`, `albums`, `movies`, `tv_series`, `video_games`, as well as various categories of `people` such as `athletes`, `musicians` and `authors`.

Downloading and parsing Wikipedia for the needed data is as simple as:

```python
from wikirec import data_utils

# Downloads the most recent stable bz2 compressed Wikipedia dump
files = data_utils.download_wiki()

# Produces an ndjson of all book articles on Wikipedia
data_utils.parse_to_ndjson(
    topic="books",
    output_path="wiki_book_articles.ndjson",
    multicore=True,
    verbose=True,
)
```

# Methods [`↩`](#jumpto)

wikirec uses natural language processing models to derive topics and then find relations between Wikipedia entries given their topic structure. Current NLP modeling methods implemented include:

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of wikirec, documents or text entries are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics.

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on nlp models ran over open source Wikipedia data. These representations are leveraged to derive topics that are then used to deliver recommendations.

### LDA with BERT embeddings

The combination of LDA with BERT via [wikirec.autoencoder](https://github.com/andrewtavis/wikirec/blob/main/wikirec/autoencoder.py).

# Usage [`↩`](#jumpto)

The following are examples of recommendations using wikirec:

```python
import wikirec
```

# To-Do [`↩`](#jumpto)

- Adding further methods for recommendations
- Allowing a user to specify multiple articles of interest
- Allowing a user to input their preference for something and then update their recommendations
- Adding support for non-English versions of Wikipedia
- Compiling other sources of open source data that can be used to augment input data
  - Potentially writing scripts to load this data for significant topics
<!---
- Creating, improving and sharing [examples](https://github.com/andrewtavis/wikirec/tree/main/examples)
--->
- Updating and refining the [documentation](https://wikirec.readthedocs.io/en/latest/)
- Improving [tests](https://github.com/andrewtavis/wikirec/tree/main/tests) for greater [code coverage](https://codecov.io/gh/andrewtavis/wikirec)

# References
<details><summary><strong>List of references<strong></summary>
<p>

- https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9

- https://towardsdatascience.com/wikipedia-data-science-working-with-the-worlds-largest-encyclopedia-c08efbac5f5c

- https://medium.com/swiftworld/topic-modeling-of-new-york-times-articles-11688837d32f

- https://blog.insightdatascience.com/news4u-recommend-stories-based-on-collaborative-reader-behavior-9b049b6724c4

</p>
</details>

# Powered By

<div align="center">
  <a href="https://www.wikipedia.org/"><img height="150" src="https://raw.githubusercontent.com/andrewtavis/wikirec/master/resources/gh_images/wikipedia_logo.png" alt="wikipedia"></a>
</div>
