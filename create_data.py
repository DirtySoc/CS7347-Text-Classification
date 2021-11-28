# %% import relevant libraries
import os
import numpy as np
import pandas as pd
import spacy

# Spacy imports
from spacy import displacy
from spacy.lang.en import English
from spacy.tokens import DocBin

# ML 
from sklearn.model_selection import train_test_split

# tqdm shows a progress bar while executing cells
from tqdm.auto import tqdm

# %% Load in data
data_path = r'IMDB Dataset.csv'
data = pd.read_csv(data_path)

data_sample = data.sample(frac=.1, random_state=1)
data_sample.value_counts(subset='sentiment')

# %% split data into train, valid, and test datasets
X_train, X_rem, y_train, y_rem = train_test_split(data_sample['review'], data_sample['sentiment'], train_size=.6)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, train_size=.5)

d = {'review': X_train, 'sentiment': y_train}
train_df = pd.DataFrame(data=d)
del d

d = {'review': X_valid, 'sentiment': y_valid}
valid_df = pd.DataFrame(data=d)
del d

d = {'review': X_test, 'sentiment': y_test}
test_df = pd.DataFrame(data=d)
del d

# %% confirm balance of categories for each dataframe

print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)

print(train_df.value_counts(subset='sentiment'))
print(valid_df.value_counts(subset='sentiment'))
print(test_df.value_counts(subset='sentiment'))

# format data for spaCy
train_df = [tuple(x) for x in train_df.to_numpy()]
valid_df = [tuple(x) for x in valid_df.to_numpy()]
test_df = [tuple(x) for x in test_df.to_numpy()]

# %% make_docs takes in data and converts it to a spaCy document
def make_docs(data):
    docs = []

    for doc, label in tqdm(nlp.pipe(data, as_tuples=True), total = len(data)):
        if label =="negative":
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
        else:
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
        docs.append(doc)

    return docs

# %% Load the pre-trained spaCy model for the english language
nlp = spacy.load("en_core_web_sm")

# %% create spaCy docbins and save to disk
import os
if not os.path.exists('data'):
    os.makedirs('data')

train_docs = make_docs(train_df)
doc_bin = DocBin(docs=train_docs)
doc_bin.to_disk("./data/train.spacy")

valid_docs = make_docs(valid_df)
doc_bin = DocBin(docs=valid_docs)
doc_bin.to_disk("./data/valid.spacy") 

# %%
test_docs = make_docs(test_df)
doc_bin = DocBin(docs=test_docs)
doc_bin.to_disk("./data/test.spacy") 

# %%
