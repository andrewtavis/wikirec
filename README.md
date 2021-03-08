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
[![codestyle](https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252)](https://colab.research.google.com/github/andrewtavis/wikirec)

### Multilingual recommendation engines based on Wikipedia data

[//]: # "The '-' after the section links is needed to make them work on GH (because of ↩s)"
**Jump to:**<a id="jumpto"></a> [Data](#data-) • [Methods](#methods-) • [Recommendations](#recommendations-) • [Comparative Results](#comparative-results-) • [To-Do](#to-do-)

**wikirec** is a framework that allows users to parse Wikipedia in any language for entries of a given type and then seamlessly create recommendation engines based on unsupervised natural language processing. The gaol is for wikirec to both refine and deploy models that provide accurate content recommendations based solely on open-source data.

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

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) allows a user to download Wikipedia articles of a given document topic including movies, TV shows, books, music, and countless other classes of information. These texts then serve as the basis to recommend similar content given an input of what the user is interested in.

Article topics are derived from infobox types found on Wikipedia articles. The [article on infoboxes](https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes) (and its translations) contains all the allowed arguments to subset the data by. Simply passing `"Infobox chosen_type"` to the `topic` argument of `data_utils.parse_to_ndjson()` in the following example will subset all Wikipedia articles for the given type. For the English Wikipedia, wikirec also provides concise arguments for data that commonly serve as recommendation inputs including: `books`, `songs`, `albums`, `movies`, `tv_series`, `video_games`, as well as various categories of `people` such as `athletes`, `musicians` and `authors`.

Downloading and parsing Wikipedia for the needed data is as simple as:

```python
from wikirec import data_utils

# Downloads the most recent stable bz2 compressed English Wikipedia dump
files = data_utils.download_wiki(language="en")

# Produces an ndjson of all book articles on Wikipedia
data_utils.parse_to_ndjson(
    topic="books",
    output_path="enwiki_books.ndjson",
    limit=None, # articles per file to find
    multicore=True,
    verbose=True,
)
```

The [examples](https://github.com/andrewtavis/wikirec/tree/main/examples) directory has a compressed copy of `enwiki_books.ndjson` for testing purposes.

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) also provides a standardized multilingual cleaning process for the loaded articles. See [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/wikirec/languages.py) for a full breakdown of what is available for each language. Generating a clean text corpus is achieved through the following:

```python
import json

with open("enwiki_books.ndjson", "r") as f:
    books = [json.loads(l) for l in f]

titles = [b[0] for b in books]
texts = [b[1] for b in books]

text_corpus, selected_idxs = data_utils.clean(
    texts=texts,
    language="en",
    min_token_freq=5,
    min_token_len=3,
    min_tokens=50,
    max_token_index=-1,
)

selected_titles = [titles[i] for i in selected_idxs]
```

# Methods [`↩`](#jumpto)

Recommendations in wikirec are generated from similarity matrices derived from trained models. The matrices represent article-article `cosine` or `euclidean` similarities that can then be sorted and selected from. Implemented NLP modeling methods within [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/wikirec/model.py) include:

### BERT

[Bidirectional Encoder Representations from Transformers](https://github.com/google-research/bert) derives representations of words based on NLP models ran over open source Wikipedia data. These representations are leveraged to derive article similarities that are then used to deliver recommendations.

```python
from wikirec import model

# We can pass kwargs for sentence_transformers.SentenceTransformer.encode
bert_embeddings = model.gen_embeddings(
        method="bert",
        corpus=text_corpus,
        bert_st_model="xlm-r-bert-base-nli-stsb-mean-tokens",
        batch_size=32,
)
bert_sim_matrix = model.gen_sim_matrix(
        method="bert",
        metric="cosine",  # euclidean
        embeddings=bert_embeddings,
)
```

### Doc2vec

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
doc2vec_sim_matrix = model.gen_sim_matrix(
    method="doc2vec",
    metric="cosine",  # euclidean
    embeddings=d2v_embeddings,
)
```

### LDA

[Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. In the case of wikirec, Wikipedia articles are posited to be a mixture of a given number of topics, and the presence of each word in a text body comes from its relation to these derived topics. These topic-word relations are then used to determine article similarities and then make recommendations.

```python
from wikirec import model

# We can pass kwargs for gensim.models.ldamulticore.LdaMulticore
lda_embeddings = model.gen_embeddings(
        method="lda",
        corpus=text_corpus,
        num_topics=35,
        passes=10,
        decay=0.5,
)
lda_sim_matrix = model.gen_sim_matrix(
    method="lda",
    metric="cosine",  # euclidean not an option at this time
    embeddings=lda_embeddings,
)
```

### TFIDF

[Term Frequency Inverse Document Frequency](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. In case of wikirec, word importances are combined and compared to derive article similarities and thus provide recommendations.

```python
from wikirec import model

# We can pass kwargs for sklearn.feature_extraction.text.TfidfVectorizer
tfidf_embeddings = model.gen_embeddings(
        method="tfidf",
        corpus=text_corpus,
)
tfidf_sim_matrix = model.gen_sim_matrix(
    method="tfidf",
    metric="cosine",  # euclidean
    embeddings=tfidf_embeddings,
)
```

# Recommendations [`↩`](#jumpto)

Once a similarity matrix for any of the above methods has been created, generating recommendations is as simple as the following:

```python
from wikirec import model

recs = model.recommend(
    inputs="title_or_list_of_titles",
    titles=selected_titles,
    sim_matrix=chosen_sim_matrix,
    n=10,
)
```

# Comparative Results [`↩`](#jumpto)

TFIDF generally outperformed all other methods in terms of providing what the user would expect, with the results being all the more striking considering its runtime is by far the shortest. The other strong performing model is BERT, as it does the best job of providing novel but sensible recommendations. LDA with the second shortest runtime provides novel recommendations along with what is expected, but recommends things that seem out of place more often than BERT. Doc2vec performs very poorly in that most results are nonsense, and it further takes the longest to train.

See [examples/rec_books](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_books.ipynb) and [examples/rec_movies](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_movies.ipynb) for fully detailed usage with model comparisons, or open these notebooks in [Google Colab](https://colab.research.google.com/github/andrewtavis/wikirec) to experiment yourself.

A sample of TFIDF and BERT book recommendations using cosine similarity follows:

```
--TFIDF--
Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Chamber of Secrets', 0.6644434544958339],
 ['Harry Potter and the Deathly Hallows', 0.6547119201881699],
 ['Harry Potter and the Order of the Phoenix', 0.642515197024424],
 ['Harry Potter and the Half-Blood Prince', 0.6420367483506718],
 ['Harry Potter and the Goblet of Fire', 0.5951892674032896],
 ['The Magical Worlds of Harry Potter', 0.5609077339791474],
 ['Harry Potter and the Methods of Rationality', 0.5379869646546271],
 ['Harry, A History', 0.5360108950074289],
 ['Harry Potter and the Cursed Child', 0.5019619677408425],
 ['Fantastic Beasts and Where to Find Them', 0.48860861798783956]]

 Note: The Prisoner of Azkaban is lost due to the prevalence of 'black'

The Hobbit recommendations:
 [['The History of The Hobbit', 0.6968314428704283],
 ['The Annotated Hobbit', 0.6056439862168547],
 ['The Lord of the Rings', 0.5579023075664765],
 ['The Road to Middle-Earth', 0.5376868352509242],
 ['Mr. Bliss', 0.49951465923170035],
 ['The Letters of J. R. R. Tolkien', 0.49586730946969954],
 ['J. R. R. Tolkien: A Biography', 0.491002699969649],
 ['The Fellowship of the Ring', 0.4725626366254413],
 ['Tolkien: Maker of Middle-earth', 0.46079070209443573],
 ["Bilbo's Last Song", 0.45336292390290156]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
 [['The History of The Hobbit', 0.3630409561901944],
 ['Harry Potter and the Chamber of Secrets', 0.35662779096968417],
 ['Harry Potter and the Deathly Hallows', 0.3525511840389999],
 ['Harry Potter and the Half-Blood Prince', 0.34751621204943556],
 ['Harry Potter and the Order of the Phoenix', 0.3429859797299906],
 ['Harry Potter and the Goblet of Fire', 0.32341644906938327],
 ['The Annotated Hobbit', 0.3146736587107837],
 ['The Lord of the Rings', 0.3108195325708887],
 ['The Magical Worlds of Harry Potter', 0.308962426272421],
 ['Harry Potter and the Methods of Rationality', 0.28898077558075325]]

 --BERT--
 Harry Potter and the Philosopher's Stone recommendations:
 [['Harry Potter and the Goblet of Fire', 0.9275407],
 ['Harry Potter and the Deathly Hallows', 0.92178226],
 ['A Monster Calls', 0.9148517],
 ['Spells', 0.9139519],
 ['Matilda', 0.9071869],
 ['Wildwood', 0.9070556],
 ['The Hobbit', 0.9052026],
 ['Harry Potter and the Order of the Phoenix', 0.9049706],
 ["The Magician's Nephew", 0.9039968],
 ['The Silver Chair', 0.89989555]]

 The Hobbit recommendations:
 [['The Seeing Stone', 0.9184302],
 ['Charmed Life', 0.9156177],
 ['Spellbound', 0.9137267],
 ['The Little Grey Men', 0.91196114],
 ['The Ring of Solomon', 0.909929],
 ['The Magic Finger', 0.9097778],
 ['I, Coriander', 0.90645945],
 ["Harry Potter and the Philosopher's Stone", 0.9052026],
 ["All Thirteen: The Incredible Cave Rescue of the Thai Boys' Soccer Team",
  0.9048355],
 ['Miss Hickory', 0.9041687]]

 Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
 [['The Little Grey Men', 0.9031571],
 ['The Magic Finger', 0.90149724],
 ['Matilda', 0.9011334],
 ['The Seeing Stone', 0.90090525],
 ['A Monster Calls', 0.9001728],
 ['Spells', 0.89896786],
 ['Charmed Life', 0.89813614],
 ["The Magician's Nephew", 0.896847],
 ['The Lion, the Witch and the Wardrobe', 0.8954387],
 ['I, Coriander', 0.8934685]]
```

# To-Do [`↩`](#jumpto)

- Adding and refining methods for recommendations in [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/wikirec/model.py)
- Creating, improving and sharing [examples](https://github.com/andrewtavis/wikirec/tree/main/examples)
- Adding methods to analyze model performance and recommendation accuracy
- Compiling other sources of open-source data that can be used to augment input data
  - Potentially writing scripts to load this data for significant topics
- Allowing multiple infobox topics to be subsetted for at once in [wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py) functions
- Speeding up the cleaning process of [wikirec.data_utils.clean()](https://github.com/andrewtavis/wikirec/blob/main/wikirec/data_utils.py)
- Updates to [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/wikirec/languages.py) as lemmatization and other linguistic package dependencies evolve
- Making recommendations faster in [wikirec.model.recommend()](https://github.com/andrewtavis/wikirec/blob/main/wikirec/model.py)
  - Recommendations can at times take a moment given the for loop to find input names
  - This could be sped up by converting sim_matrix instances into an indexed data type within `wikirec.model.recommend()` (on average would be faster)
- Allowing euclidean distance measurements for LDA based recommendations in [wikirec.model.gen_sim_matrix()](https://github.com/andrewtavis/wikirec/blob/main/wikirec/model.py)
- Expanding the [documentation](https://wikirec.readthedocs.io/en/latest/)
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
