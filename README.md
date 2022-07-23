<div align="center">
  <a href="https://github.com/andrewtavis/wikirec"><img src="https://raw.githubusercontent.com/andrewtavis/wikirec/main/.github/resources/logo/wikirec_logo_transparent.png" width="529" height="169"></a>
</div>

<ol></ol>

[![rtd](https://img.shields.io/readthedocs/wikirec.svg?logo=read-the-docs)](http://wikirec.readthedocs.io/en/latest/)
[![ci](https://img.shields.io/github/workflow/status/andrewtavis/wikirec/CI?logo=github)](https://github.com/andrewtavis/wikirec/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/andrewtavis/wikirec/branch/main/graphs/badge.svg)](https://codecov.io/gh/andrewtavis/wikirec)
[![pyversions](https://img.shields.io/pypi/pyversions/wikirec.svg?logo=python&logoColor=FFD43B&color=306998)](https://pypi.org/project/wikirec/)
[![pypi](https://img.shields.io/pypi/v/wikirec.svg?color=4B8BBE)](https://pypi.org/project/wikirec/)
[![pypistatus](https://img.shields.io/pypi/status/wikirec.svg)](https://pypi.org/project/wikirec/)
[![license](https://img.shields.io/github/license/andrewtavis/wikirec.svg)](https://github.com/andrewtavis/wikirec/blob/main/LICENSE.txt)
[![coc](https://img.shields.io/badge/coc-Contributor%20Covenant-ff69b4.svg)](https://github.com/andrewtavis/wikirec/blob/main/.github/CODE_OF_CONDUCT.md)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![colab](https://img.shields.io/badge/%20-Open%20in%20Colab-097ABB.svg?logo=google-colab&color=097ABB&labelColor=525252)](https://colab.research.google.com/github/andrewtavis/wikirec)

## Recommendation engine framework based on Wikipedia data

**wikirec** is a framework that allows users to parse Wikipedia in any language for entries of a given type and then seamlessly generate recommendations for the given content. Recommendations are based on unsupervised natural language processing over article texts, with ratings being leveraged to weigh inputs and indicate preferences. The goal is for wikirec to both refine and deploy models that provide accurate content recommendations with only open-source data.

See the [documentation](https://wikirec.readthedocs.io/en/latest/) for a full outline of the package including models and data preparation.

<a id="contents"></a>

# **Contents**

- [Installation](#installation)
- [Data](#data)
  - [Downloading](#downloading)
  - [Cleaning](#cleaning)
- [Methods](#methods)
  - [BERT](#bert)
  - [Doc2Vec](#doc2vec)
  - [LDA](#lda)
  - [TFIDF](#tfidf)
  - [WikilinkNN](#wikilinknn)
- [Recommendations](#recommendations)
- [Comparative Results](#comparative-results)
- [To-Do](#to-do)

<a id="installation"></a>

# Installation [`⇧`](#contents)

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

<a id="data"></a>

# Data [`⇧`](#contents)

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) allows a user to download Wikipedia articles of a given topic including movies, TV shows, books, music, and countless other classes of information. These texts then serve as the basis to recommend similar content given an input of what the user is interested in.

Article topics are derived from infobox types found on Wikipedia articles. The [article on infoboxes](https://en.wikipedia.org/wiki/Wikipedia:List_of_infoboxes) (and its translations) contains all the allowed arguments to subset the data by. Simply passing `"Infobox chosen_type"` to the `topics` argument of `data_utils.parse_to_ndjson()` in the following example will subset all Wikipedia articles for the given type. Lists can also be passed if more than one topic is desired. For the English Wikipedia, wikirec also provides concise arguments for data that commonly serve as recommendation inputs including: `books`, `songs`, `albums`, `movies`, `tv_series`, `video_games`, as well as various categories of `people` such as `athletes`, `musicians` and `authors` (see [data_utils.input_conversion_dict()](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py)).

Data processing in wikirec involves the following steps:

<a id="downloading"></a>

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

<a id="cleaning"></a>

<details><summary><strong>Cleaning Parsed Articles</strong></summary>
<p>

[wikirec.data_utils](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) also provides a standardized multilingual cleaning process for the parsed articles. See [wikirec.languages](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/languages.py) for a full breakdown of what is available for each language. Generating a clean text corpus is achieved through the following:

```python
import json

with open("./enwiki_books.ndjson", "r") as f:
    books = [json.loads(l) for l in f]

titles = [b[0] for b in books]
texts = [b[1] for b in books]
wikilinks = [b[2] for b in books]  # internal wikipedia links for WikilinkNN method

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

<a id="methods"></a>

# Methods [`⇧`](#contents)

Recommendations in wikirec are generated from similarity matrices derived from trained model embeddings. Implemented NLP modeling methods within [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) include:

<a id="bert"></a>

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

<a id="doc2vec"></a>

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

<a id="lda"></a>

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

<a id="tfidf"></a>

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

<a id="wikilinknn"></a>

<details><summary><strong>WikilinkNN</strong></summary>
<p>

Based on this [Towards Data Science article](https://towardsdatascience.com/building-a-recommendation-system-using-neural-network-embeddings-1ef92e5c80c9), the wikilink neural network method makes the assumption that content will be similar if they are linked to the same Wikipedia articles. A corpus of internal wikilinks per article is passed, and embeddings based on these internal references are then derived. Note that model hyperparameters are dramatically more important in this approach than in others.

```python
from wikirec import model

wikilink_embeddings = model.gen_embeddings(
        method="WikilinkNN",
        path_to_json="./enwiki_books.ndjson",  # json used instead of a corpus
        path_to_embedding_model="books_embedding_model.h5",
        embedding_size=75,
        epochs=20,
        verbose=True,
)
```

<p>
</details>

<a id="recommendations"></a>

# Recommendations [`⇧`](#contents)

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

<a id="comparative-results"></a>

# Comparative Results [`⇧`](#contents)

- TFIDF generally outperformed all other NLP methods in terms of providing what the user would expect, with the results being all the more striking considering its runtime is by far the shortest.
- The other strong performing NLP model is BERT, as it does the best job of providing novel but sensible recommendations.
- WikilinkNN also provides very sensible results, giving wikirec effective modeling options using different kinds of inputs.
- LDA with the second shortest runtime provides novel recommendations along with what is expected, but recommends things that seem out of place more often than BERT.
- Doc2vec performs very poorly in that most results are nonsense, and it further takes the longest to train.

See [examples/rec_books](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_books.ipynb) and [examples/rec_movies](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_movies.ipynb) for detailed demonstrations with model comparisons, as well as [examples/rec_ratings](https://github.com/andrewtavis/wikirec/blob/main/examples/rec_ratings.ipynb) for how to leverage user ratings. These notebooks can also be opened in [Google Colab](https://colab.research.google.com/github/andrewtavis/wikirec) for direct experimentation.

Sample recommendations for single and multiple inputs are found in the following dropdowns:

<details><summary><strong>TFIDF</strong></summary>
<p>

```_output
Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Deathly Hallows', 0.6046299758645369],
 ['Harry Potter and the Chamber of Secrets', 0.6006421605504958],
 ['Harry Potter and the Order of the Phoenix', 0.5965340424789338],
 ['Harry Potter and the Goblet of Fire', 0.5569541701616842],
 ['Harry Potter and the Half-Blood Prince', 0.5525197546210491],
 ['The Magical Worlds of Harry Potter', 0.5328091662536486],
 ['Harry Potter and the Prisoner of Azkaban', 0.491142269221778],
 ['Harry, A History', 0.461521032636577],
 ['Fantastic Beasts and Where to Find Them', 0.458905951118587],
 ['Harry Potter and the Methods of Rationality', 0.45024337149870786]]

The Hobbit recommendations:
[['The History of The Hobbit', 0.7654956800395748],
 ['The Annotated Hobbit', 0.6429102504821168],
 ['The Lord of the Rings', 0.5373413608301959],
 ['The Road to Middle-Earth', 0.5306535049915708],
 ['The Letters of J. R. R. Tolkien', 0.48933976150601666],
 ['The Marvellous Land of Snergs', 0.48317913980292654],
 ['Mr. Bliss', 0.4803612654025307],
 ['J. R. R. Tolkien: A Biography', 0.4801418285780905],
 ['A Companion to J. R. R. Tolkien', 0.4668405235491576],
 ['Tolkien: A Look Behind  " The Lord of the Rings "', 0.45164156724562365]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The History of The Hobbit', 0.39710714157986077],
 ['The Annotated Hobbit', 0.3339037084669694],
 ['Harry Potter and the Chamber of Secrets', 0.32972850299980644],
 ['Harry Potter and the Deathly Hallows', 0.32760681591732854],
 ['Harry Potter and the Order of the Phoenix', 0.319444468511931],
 ['The Lord of the Rings', 0.3069697109614444],
 ['Harry Potter and the Half-Blood Prince', 0.3022894152745786],
 ['Harry Potter and the Goblet of Fire', 0.3019957448304001],
 ['The Magical Worlds of Harry Potter', 0.2996981871702149],
 ['The Road to Middle-Earth', 0.28697680264545045]]
```

<p>
</details>

<details><summary><strong>BERT</strong></summary>
<p>

```_output
Harry Potter and the Philosopher's Stone recommendations:
[['The Magical Worlds of Harry Potter', 0.88391376],
 ['Harry Potter and the Chamber of Secrets', 0.8779844],
 ['Harry Potter and the Order of the Phoenix', 0.8671646],
 ['Harry Potter and the Prisoner of Azkaban', 0.85335326],
 ['Harry Potter and the Half-Blood Prince', 0.84942037],
 ['Harry Potter and the Goblet of Fire', 0.8481754],
 ['Year of the Griffin', 0.8280591],
 ['Magyk', 0.8277706],
 ['Harry Potter and the Deathly Hallows', 0.8257748],
 ['The Weirdstone of Brisingamen', 0.81287163]]

The Hobbit recommendations:
[['The Lord of the Rings', 0.8506559],
 ["The Shepherd's Crown", 0.84309],
 ['The War That Saved My Life', 0.8352962],
 ['The Foundling and Other Tales of Prydain', 0.8336451],
 ["The Inquisitor's Tale", 0.83097416],
 ['Ruby Holler', 0.8303863],
 ['Sam and Dave Dig a Hole', 0.82980216],
 ['Fattypuffs and Thinifers', 0.82704884],
 ['El Deafo', 0.8226619],
 ['Beast (Kennen novel)', 0.8221826]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The Weirdstone of Brisingamen', 0.8108008205890656],
 ['The Magical Worlds of Harry Potter', 0.7868899703025818],
 ["The Golem's Eye", 0.7817798852920532],
 ['Harry Potter and the Prisoner of Azkaban', 0.7784444689750671],
 ['The Last Battle', 0.7773005664348602],
 ['Child Christopher and Goldilind the Fair', 0.776639997959137],
 ["The Inquisitor's Tale", 0.7743396461009979],
 ['Charmed Life (novel)', 0.7735742926597595],
 ['A Wizard of Earthsea', 0.7710956037044525],
 ["Conrad's Fate", 0.770046204328537]]
```

<p>
</details>

<details><summary><strong>WikilinkNN</strong></summary>
<p>

```_output
Harry Potter and the Philosopher's Stone recommendations:
[['Harry Potter and the Chamber of Secrets', 0.9697026],
 ['Harry Potter and the Goblet of Fire', 0.969065],
 ['Harry Potter and the Deathly Hallows', 0.9685888],
 ['Harry Potter and the Half-Blood Prince', 0.9635748],
 ['Harry Potter and the Prisoner of Azkaban', 0.9569129],
 ['Harry Potter and the Order of the Phoenix', 0.94091964],
 ['Harry Potter and the Cursed Child', 0.9358928],
 ['My Immortal (fan fiction)', 0.91195196],
 ['Eragon', 0.89236057],
 ['Quidditch Through the Ages', 0.8891448]]

The Hobbit recommendations:
[['The Lord of the Rings', 0.94245297],
 ['The Silmarillion', 0.9160445],
 ['Beren and Lúthien', 0.90604335],
 ['The Fall of Gondolin', 0.9044683],
 ['The Children of Húrin', 0.895282],
 ['The Book of Lost Tales', 0.89020956],
 ['The Road to Middle-Earth', 0.88268256],
 ["The Magician's Nephew", 0.8816683],
 ['The History of The Hobbit', 0.87789804],
 ['Farmer Giles of Ham', 0.87786204]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The Lord of the Rings', 0.8367433249950409],
 ['Harry Potter and the Deathly Hallows', 0.8294640183448792],
 ['The Children of Húrin', 0.8240831792354584],
 ['Harry Potter and the Prisoner of Azkaban', 0.8158660233020782],
 ['Harry Potter and the Goblet of Fire', 0.8150344789028168],
 ['Eragon', 0.8118217587471008],
 ['Harry Potter and the Chamber of Secrets', 0.8101150393486023],
 ['Fantastic Beasts and Where to Find Them', 0.8092647194862366],
 ['Watership Down', 0.8012698292732239],
 ['Harry Potter and the Half-Blood Prince', 0.7979166805744171]]

```

<p>
</details>

<details><summary><strong>Weighted Model Approach</strong></summary>
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
[['Harry Potter and the Chamber of Secrets', 0.7809146131987466],
 ['Harry Potter and the Order of the Phoenix', 0.7724439006273619],
 ['The Magical Worlds of Harry Potter', 0.7610271015260268],
 ['Harry Potter and the Deathly Hallows', 0.7483740864279236],
 ['Harry Potter and the Goblet of Fire', 0.746247955871592],
 ['Harry Potter and the Half-Blood Prince', 0.7455051626944851],
 ['Harry Potter and the Prisoner of Azkaban', 0.7265793668098672],
 ['Harry Potter and the Cursed Child', 0.6773072534713512],
 ['Harry, A History', 0.6772576164353141],
 ['Fantastic Beasts and Where to Find Them', 0.626084297475856]]

The Hobbit recommendations:
[['The Lord of the Rings', 0.7409957782467453],
 ['The History of The Hobbit', 0.7352996903587457],
 ['The Annotated Hobbit', 0.7135948210557342],
 ['The Marvellous Land of Snergs', 0.6838799880927064],
 ['The Road to Middle-Earth', 0.6447863856578011],
 ['The Silmarillion', 0.6445419659298917],
 ['A Companion to J. R. R. Tolkien', 0.6416663828729424],
 ['J. R. R. Tolkien: A Biography', 0.6347377961302614],
 ['The Children of Húrin', 0.6261937795502842],
 ['Mr. Bliss', 0.6217533139998945]]

Harry Potter and the Philosopher's Stone and The Hobbit recommendations:
[['The Magical Worlds of Harry Potter', 0.6163728193841632],
 ['Harry Potter and the Order of the Phoenix', 0.6098655072975429],
 ['Harry Potter and the Prisoner of Azkaban', 0.6026408288502743],
 ['Harry Potter and the Chamber of Secrets', 0.5966943180957163],
 ['Harry Potter and the Deathly Hallows', 0.5932562267661715],
 ['The Lord of the Rings', 0.5931736380571248],
 ['Harry Potter and the Half-Blood Prince', 0.5905134043157909],
 ['The Weirdstone of Brisingamen', 0.5620134317676433],
 ['Fantastic Beasts and Where to Find Them', 0.5594706076813922],
 ['Harry Potter and the Goblet of Fire', 0.556541219039868]]
```

The WikilinkNN model can be combined with other models by subsetting the similarity matrix for titles derived in the cleaning process:

```python
wikilink_sims_copy = wikilink_sims.copy()
not_selected_idxs = [i for i in range(len(titles)) if i not in selected_idxs]

wikilink_sims_copy = np.delete(wikilink_sims_copy, not_selected_idxs, axis=0)
wikilink_sims_copy = np.delete(wikilink_sims_copy, not_selected_idxs, axis=1)
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

<a id="to-do"></a>

# To-Do [`⇧`](#contents)

Please see the [contribution guidelines](https://github.com/andrewtavis/wikirec/blob/main/.github/CONTRIBUTING.md) if you are interested in contributing to this project. Work that is in progress or could be implemented includes:

- Allowing a user to express disinterest in [wikirec.model.recommend](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) [(see issue)](https://github.com/andrewtavis/wikirec/issues/33)

- Devising methods to best combine recommendations for more than one input [(see issue)](https://github.com/andrewtavis/wikirec/issues/32)

- Adding arguments to [data_utils.input_conversion_dict()](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/data_utils.py) based on Wikipedia languages to simplify parsing arguments [(see issue)](https://github.com/andrewtavis/wikirec/issues/34)

- Adding and refining methods for recommendations in [wikirec.model](https://github.com/andrewtavis/wikirec/blob/main/src/wikirec/model.py) [(see issue)](https://github.com/andrewtavis/wikirec/issues/31)

- Adding embeddings visualization methods to wikirec [(see issue)](https://github.com/andrewtavis/wikirec/issues/35)

- Allowing a user to use article topics to further express preferences in recommendations [(see issue)](https://github.com/andrewtavis/wikirec/issues/54)

- Integrating metadata from Wikidata [(see issue)](https://github.com/andrewtavis/wikirec/issues/55)

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
  <br>
  <a href="https://www.wikipedia.org/"><img height="150" src="https://raw.githubusercontent.com/andrewtavis/wikirec/main/.github/resources/images/wikipedia_logo.png" alt="Wikipedia"></a>
  <br>
</div>
