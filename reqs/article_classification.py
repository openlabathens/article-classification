#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import spacy
from nltk.corpus import stopwords
# nltk.download('stopwords')
from collections import Counter
from sklearn import model_selection, feature_extraction, metrics
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import seaborn as sns
sns.set()


# Import data set:
# 

# In[21]:


pd.options.mode.chained_assignment = None

col_list = ['text', 'class']
df_full = pd.read_csv('data/data_set.csv', usecols=col_list)
df = df_full[['text']]

df["text"] = df["text"].astype(str)
df_full.head()
df.head()


# # Text preprocessing:
# 

# ## lower casing:
# 

# In[22]:



df["text_lower"] = df["text"].str.lower()
df.head()
# drop the new column created in last cell
# df.drop(["text_lower"], axis=1, inplace=True)


# ## remove punctuation
# 

# In[23]:


punctuation = '«!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~»●·’“”'
PUNCT_TO_REMOVE = punctuation


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


df["text_wo_punct"] = df["text_lower"].apply(
    lambda text: remove_punctuation(text))
df.head()


# ## remove stopwords:
# 

# In[24]:


STOPWORDS_GREEK = set(stopwords.words('greek'))


def import_additional_greek_stopwords(STOPWORDS_GREEK):
    STOPWORDS_GREEK.add('της')
    STOPWORDS_GREEK.add('από')
    STOPWORDS_GREEK.add('είναι')
    STOPWORDS_GREEK.add('έχει')
    STOPWORDS_GREEK.add('σας')
    STOPWORDS_GREEK.add('τους')
    STOPWORDS_GREEK.add('τη')
    STOPWORDS_GREEK.add('μας')
    STOPWORDS_GREEK.add('στα')
    STOPWORDS_GREEK.add('στις')
    STOPWORDS_GREEK.add('στους')
    STOPWORDS_GREEK.add('μου')
    STOPWORDS_GREEK.add('σου')
    return STOPWORDS_GREEK


STOPWORDS_GREEK = import_additional_greek_stopwords(STOPWORDS_GREEK)


def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS_GREEK])


df["text_wo_stop"] = df["text_wo_punct"].apply(
    lambda text: remove_stopwords(text))
df.head()


# ## remove intonation:
# 

# In[25]:


def remove_intonation(text):

    rep = {"ά": "α", "έ": "ε", "ή": "η", "ί": "ι", "ό": "ο", "ύ": "υ", "ώ": "ω", "ϊ": "ι",
           "ἀ": "α", "ἐ": "ε", "ἤ": "η", "ἰ": "ι", "ἄ": "α", "ὐ": "υ", "ὡ": "ω", "ὦ": "ω",
           'ὖ': 'υ', 'ὅ': 'ο', 'ῆ': 'η', 'ῇ': 'η', 'ῦ': 'υ', 'ὁ': 'ο', 'ὑ': 'υ', 'ὲ': 'ε',
           'ὺ': 'υ', 'ἂ': 'α', 'ἵ': 'ι', 'ὴ': 'η', 'ὰ': 'α', 'ἅ': 'α', 'ὶ': 'ι', 'ἴ': 'ι',
           'ὸ': 'ο', 'ἥ': 'η', 'ἡ': 'η', 'ὕ': 'υ', 'ἔ': 'ε', 'ἳ': 'ι', 'ὗ': 'υ', 'ἃ': 'α',
           'ὃ': 'ο', 'ὥ': 'ω', 'ὔ': 'υ', 'ῖ': 'ι', 'ἣ': 'η', 'ἷ': 'ι', 'ἑ': 'ε', 'ᾧ': 'ω',
           'ἢ': 'η', 'ΐ': 'ι', }

    rep = dict((nltk.re.escape(k), v) for k, v in rep.items())
    pattern = nltk.re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[nltk.re.escape(m.group(0))], text)

    return text


df["text_wo_intonation"] = df["text_wo_stop"].apply(
    lambda text: remove_intonation(text))
df.head()


# ## remove frequent words:
# 

# ### get most frequent words:
# 

# In[26]:


cnt = Counter()
for text in df["text_wo_intonation"].values:
    for word in text.split():
        cnt[word] += 1

# show ten more frequent elements:
cnt.most_common(10)


# ### remove most frequent words:
# 

# In[27]:


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])


def remove_freqwords(text):
    """custom function to remove frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])


df["text_wo_freq"] = df["text_wo_intonation"].apply(
    lambda text: remove_freqwords(text))
df.head()


# ## remove most rare words:
# 

# In[28]:


for text in df["text_wo_intonation"].values:
    for word in text.split():
        cnt[word] += 1

# show ten least frequent elements:
cnt.most_common()[:-10-1:-1]


# In[29]:


# Drop the two columns which are no longer needed
df.drop(["text_wo_punct", "text_wo_stop", "text_lower"], axis=1, inplace=True)

n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])


def remove_rarewords(text):
    """custom function to remove rare words"""
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


df["text_wo_rare"] = df["text_wo_freq"].apply(
    lambda text: remove_rarewords(text))
df.head()


# ## Lemmatization:
# 

# In[30]:


# Drop the columns which are no longer needed
df.drop(["text_wo_intonation", "text_wo_freq"], axis=1, inplace=True)

nlp = spacy.load("el_core_news_sm")
# nlp.remove_pipe("tagger")


def lemmatize_words(text):
    """custom function to lemmatize text"""
    doc = nlp(text)
    # pos_tagged_text = text.pos
    return " ".join([token.lemma_ for token in doc])


df["text_lemmatized"] = df["text_wo_rare"].apply(
    lambda text: lemmatize_words(text))
df.head()


# ## Remove numbers:
# 

# In[31]:


def drop_numbers(text):
    text_wo_numbers = re.sub(r'[0-9]+', '', text)
    return text_wo_numbers


df["text_wo_numbers"] = df["text_lemmatized"].apply(
    lambda text: drop_numbers(text))
df.head()


# ## Remove single letter words:
# 

# In[32]:


def drop_single_letter_words(text):
    return ' '.join([w for w in text.split() if len(w) > 1])


df["text_wo_single_letters"] = df["text_wo_numbers"].apply(
    lambda text: drop_single_letter_words(text))
df.head()


# ## Add labels to the pre-processed df:
# 

# In[33]:


# Drop the columns which are no longer needed
df.drop(["text", "text_wo_rare", "text_lemmatized",
        "text_wo_numbers"], axis=1, inplace=True)

# Set up data set with preprocessed text & classes:
df['label'] = df_full['class']
df.columns = ['text', 'label']
df.head()


# ## Classification Analysis:
# 

# In[63]:


label_distribution = (df['label'].value_counts() * 100) / len(df)

# Add value labels


def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], "%.2f" % round(y[i], 2) + "%", ha='center')


plt.bar(label_distribution.index, label_distribution)
add_labels(label_distribution.index, label_distribution)
plt.xlabel("Classes")
plt.ylabel("Percentage")
plt.show()

# See missing values:
df.isna().sum()


# ## Split data set to train, validate and test:
# 

# In[35]:


# df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=25)

# Split to train validate and test
df_train, df_validate, df_test = np.split(df.sample(frac=1, random_state=42), [
                                          int(.6*len(df)), int(.8*len(df))])

print(f"No. of training examples: {df_train.shape[0]}")
print(f"No. of testing examples: {df_test.shape[0]}")
print(f"No. of validating examples: {df_validate.shape[0]}")

# Get X_train, y_train, X_val, y_val, X_test, y_test

X_train = df_train['text']
y_train = df_train['label']

X_val = df_validate['text']
y_val = df_validate['label']

X_test = df_test['text']
y_test = df_test['label']


# ## TfIdfVectorizer:
# 

# In[39]:


vectorizer = TfidfVectorizer()

# remember to use the original X_train set
X_train_tfidf = vectorizer.fit_transform(X_train)
X_train_tfidf.shape


# <font color=green>This shows that training set is comprised of 310 documents, and 15186 features.</font>
# 

# ## Build some Pipelines:
# 

# ### Naive Bayes:
# 

# In[68]:


text_clf_NB = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf_NB.fit(X_train, y_train)

predicted = text_clf_NB.predict(X_val)
np.mean(predicted == y_val)

print(metrics.classification_report(y_val, predicted))

# Confusion Matrix:
cfmtx_NB = pd.DataFrame(metrics.confusion_matrix(y_val, predicted), index=[
    'Ανθρωποκτονία', 'Γυναικοκτονία'], columns=['Ανθρωποκτονία', 'Γυναικοκτονία'])

cfmtx_NB

sns.heatmap(cfmtx_NB, annot=True, fmt='d', cmap='YlGnBu')


# ### Linear Support Vector Machines:
# 

# In[69]:


text_clf_SVM = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf_SVM.fit(X_train, y_train)

predicted = text_clf_SVM.predict(X_val)
np.mean(predicted == y_val)

print(metrics.classification_report(y_val, predicted))

# Confusion Matrix:
cfmtx_SVM = pd.DataFrame(metrics.confusion_matrix(y_val, predicted), index=[
    'Ανθρωποκτονία', 'Γυναικοκτονία'], columns=['Ανθρωποκτονία', 'Γυναικοκτονία'])

cfmtx_SVM

sns.heatmap(cfmtx_SVM, annot=True, fmt='d', cmap='YlGnBu')

