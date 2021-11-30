# Movie Review Classification with spaCy

Welcome! This repo contains some files to get a spaCy model working for classifying movie reviews.

## Prerequisites

- A python env setup with the following installed:
    - spaCy
    - numpy
    - pandas
    - scikit-learn
    - tqdm

## Setup

1. Download data from Kaggle for the [IMDB 50k Movie Reviews Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv).
2. Place the CSV file in the root folder of this repo and rename it `IMDB Dataset.csv`
3. Preprocess the data with `create_data.py`
3. Create the spaCy configuration:
    1. Generate a base config for English with only textcat selected from https://spacy.io/usage/training#quickstart 
    2. Copy the configuration to a file called base_config in the root of the cloned repo.
    3. Edit the train data and dev data fields in the config so that they reference the spacy docbin files generated in step 3.
5. Generate full config, train, and evaluate the model:

```bash
# generate the full spacy config file
python -m spacy init fill-config ./base_config.cfg ./config.cfg

# train spacy model and store in output dir
python -m spacy train config.cfg --output ./output 

# evaluate trained model
python -m spacy evaluate ./output/model-best/ ./data/test.spacy -o metrics.json
```