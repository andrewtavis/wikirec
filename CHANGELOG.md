### wikirec 0.2.2 (May 18, 2021)

Changes include:

- The WikilinkNN model has been added allowing users to derive recommendations based which articles are linked to the same other Wikipedia articles
- Examples have been updated to reflect this new model
- books_embedding_model.h5 is provided for quick experimentation
- enwiki_books.ndjson has been updated with a more recent dump
- Function docstring grammar fixes
- Baseline testing for the new model has been added to the CI

### wikirec 0.2.1 (April 29, 2021)

Changes include:

- Support has been added for gensim 3.8.x and 4.x
- Wikipedia links are now an output of data_utils.parse_to_ndjson
- Dependencies in requirement and environment files are now condensed

### wikirec 0.2.0 (April 16, 2021)

Changes include:

- Users can now input ratings to weigh recommendations
- Fixes for how multiple inputs recommendations were being calculated
- Switching over to an src structure
- Code quality is now checked with Codacy
- Extensive code formatting to improve quality and style
- Bug fixes and a more explicit use of exceptions
- More extensive contributing guidelines

### wikirec 0.1.1.7 (March 14, 2021)

Changes include:

- Multiple Infobox topics can be subsetted for at the same time
- Users have greater control of the cleaning process
- The cleaning process is verbose and uses multiprocessing
- The workflow for all models has been improved and explained
- Methods have been developed to combine modeling techniques for better results

### wikirec 0.1.0 (March 8, 2021)

First stable release of wikirec

- Functions to subset Wikipedia in any language by infobox topics have been provided
- A multilingual cleaning process that can clean texts of any language to varying degrees of efficacy is included
- Similarity matrices can be generated from embeddings using the following models:
  - BERT
  - Doc2vec
  - LDA
  - TFIDF
- Similarity matrices can be created using either cosine or euclidean relations
- Usage examples have been provided for multiple input types
- Optimal LDA topic numbers can be inferred graphically
- The package is fully documented
- Virtual environment files are provided
- Extensive testing of all modules with GH Actions and Codecov has been performed
- A code of conduct and contribution guidelines are included
