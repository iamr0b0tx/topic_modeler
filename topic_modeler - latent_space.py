#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# ### imports and utilities

# In[2]:


from collections import Counter

from utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from scipy.special import softmax
from scipy.stats import norm
from scipy.stats import entropy as calculate_entropy


from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# ### load dataset

# In[3]:


# total number of samples needed
randomize = False

# retrieve dataset
categories = ['rec.autos', 'talk.politics.mideast', 'alt.atheism', 'sci.space']

docs = fetch_20newsgroups(subset='train', shuffle=randomize, remove=('headers', 'footers', 'quotes'), categories=categories)
docs, old_labels, classes = docs.data, docs.target, docs.target_names


# ### clean dataset

# In[4]:


datasize = 100
max_document_length = None

index = -1
train_docs, labels = [], []

sizes = [0]*len(categories)

with tqdm(total=len(categories)*datasize) as pbar:
    while sum(sizes) != len(categories)*datasize:
        index += 1
        size_index = categories.index(classes[old_labels[index]])
        
        if sizes[size_index] == datasize:
            continue
        
        doc = docs[index]
        status, doc, word_count = clean_doc(doc, True)
        
        if (not status) or (max_document_length is not None and len(doc) > max_document_length):
            continue
        
        labels.append(categories[size_index])
        train_docs.append(doc)
        sizes[size_index] += 1
        pbar.update(1)

labels = np.array(labels)


# In[5]:


index = 0
print(f"Topic: {labels[index]}\n{'='*50}\n{train_docs[index]}")


# In[6]:


print(sizes)
assert min(sizes) == max(sizes) == datasize


# In[7]:


print(f"there are {len(train_docs)} docs")


# ### Initialize Vectorizer

# In[8]:


# initialize the count vectorizer
vectorizer = CountVectorizer()

# fit it to dataset
vectorizer.fit(train_docs)

vocabulary = vectorizer.get_feature_names()
print("word_count is", len(vocabulary))


# ### Prepare Datatset

# In[9]:


# create doc count vectors
train_doc_vectors = vectorizer.transform(train_docs).toarray()

total_num_of_documents = len(train_doc_vectors)
print(f"{total_num_of_documents} train_docs")


# ### Word-Word Ratio

# In[10]:


# reduce freq in doc to bin value of 1 or 0
word_freq_in_doc = pd.DataFrame(train_doc_vectors, columns=vocabulary)
word_word_co = pd.DataFrame(data=0.0, columns=vocabulary, index=vocabulary)

word_doc_frequency = (word_freq_in_doc > 0).astype(int)
probability = word_doc_frequency.sum(0) / len(train_doc_vectors)

for word in tqdm(vocabulary):
    pxy = word_doc_frequency[word_doc_frequency[word] == 1].sum(0) / total_num_of_documents
#     word_word_co[word] = pxy / (probability[word] * probability)
#     word_word_co[word][word_word_co[word] > 0] = word_word_co[word][word_word_co[word] > 0]**-1
    word_word_co[word] = np.nan_to_num(sigmoid(np.nan_to_num(np.log2(pxy / (probability[word] * probability)))))

# word_word_co = (word_word_co.T / word_word_co.sum(1)).T
print(f"word_word_co has shape {word_word_co.shape}")


# In[11]:


word_word_co.head()


# ### Calculate Word Trust ratio

# In[12]:


word_entropy = pd.DataFrame(data=np.nan_to_num(calculate_entropy(word_word_co.T, base=2)), columns=[0], index=vocabulary)[0]
word_trust_factor = pd.DataFrame(data=gaussian(abs(word_entropy - word_entropy.mean())), columns=[0], index=vocabulary)[0]
word_trust_factor = word_trust_factor / word_trust_factor.max()


# In[13]:


words = ["to", "the", "algorithm", "program", "and"]
# words = np.array(vocabulary)[np.random.randint(len(vocabulary), size=5)]

fig = plt.figure(figsize=(15,3))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.set_title(f"Word Trust factor")
ax1.bar(words, word_trust_factor[words])

ax2.set_title(f"Entropy")
ax2.bar(words, word_entropy[words])

ax3.set_title(f"Word Frequency Normalized")
ax3.bar(words, probability[words])

plt.show()


# ### Observe word_word_co ratios

# In[14]:


wwc = (word_word_co * word_trust_factor)


# In[15]:


word = "software"

fig = plt.figure(figsize=(10,3))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

print(f"=== Ploting {word} against {words} ===")

values = word_word_co.loc[word][words]
ax1.set_title(f"word_word_co")
ax1.set_ylim(0, 1.5*values.max())
ax1.bar(words, values)

values = wwc.loc[word][words]
ax2.set_title(f"updated word_word_co")
ax2.set_ylim(0, 1.5*values.max())
ax2.bar(words, values)

plt.show()


# In[16]:


wwc.loc["war"].sort_values(ascending=False).head(10)


# In[17]:


word_word_co.loc["war"].sort_values(ascending=False).head(10)


# In[18]:


word_word_co = wwc


# ### Update word_word_co with word_word_co

# In[19]:


doc_word_distr = word_doc_frequency * word_trust_factor
# doc_word_distr = (doc_word_distr.T / doc_word_distr.sum(1)).T.fillna(0)


# In[20]:


doc_index = 13
word_doc_frequency.iloc[doc_index].sort_values(ascending=False).head(10)


# In[21]:


doc_word_distr.iloc[doc_index].sort_values(ascending=False).head(10)


# In[22]:


print(f"Topic: {labels[doc_index]}\n{'='*50}\n{train_docs[doc_index]}")


# In[23]:


doc_word_distr.head()


# In[24]:


for di in range(len(doc_word_distr.index)):
    print(doc_word_distr.iloc[di].sort_values(ascending=False).head(5).index.to_list())
    break


# ### Define Latent partitions

# In[241]:


# reduction = None
# reduction = "pca"
reduction = "normal"

if reduction is None:
    columns = doc_word_distr.columns
    param_values = doc_word_distr.values

if reduction == "pca":
    num_of_components = 8
    columns = list(range(num_of_components))
    
    pca = PCA(n_components=num_of_components)
    param_values = pca.fit_transform(doc_word_distr)

if reduction == "normal":
    columns = ["mean", "std"]
    column_values = [doc_word_distr.mean(1), doc_word_distr.std(1)]
    param_values = np.array(column_values).T
    
distr_params = pd.DataFrame(data=param_values, columns=columns, index=list(range(len(doc_word_distr))))
print(f"distr_params has shape {distr_params.shape}")


# In[242]:


distr_params.head()


# ### Using Kmeans MiniBatch

# In[243]:


num_of_topics = 4


# In[288]:


doc_entropy = pd.DataFrame(data=np.nan_to_num(calculate_entropy(doc_word_distr.T, base=2)), columns=[0], index=distr_params.index)[0]

# doc_trust_factor = pd.DataFrame(data=gaussian2(doc_entropy), columns=[0], index=distr_params.index)[0]
doc_trust_factor = pd.DataFrame(data=gaussian2(abs(doc_entropy - doc_entropy.mean())), columns=[0], index=distr_params.index)[0]

doc_trust_factor = doc_trust_factor / doc_trust_factor.sum()


# In[289]:


doc_trust_factor.sort_values(ascending=False)


# In[290]:


doc_word_distr.loc[263].sort_values(ascending=False).head(10)


# In[334]:


kmeans_model = KMeans(n_clusters=num_of_topics, random_state=0).fit(distr_params)
kmeans_model = KMeans(n_clusters=num_of_topics, random_state=0).fit(distr_params, sample_weight=distr_params.std(1))
# kmeans_model = KMeans(n_clusters=num_of_topics, random_state=0).fit(distr_params, sample_weight=doc_trust_factor)


# In[335]:


# kmeans_model = MiniBatchKMeans(n_clusters=num_of_topics, random_state=0)

# num_of_iterations = 256

# num_of_samples = len(distr_params)
# batch_size = num_of_samples // 2

# for i in tqdm(range(num_of_iterations)):
#     indices = np.random.randint(num_of_samples, size=batch_size)
    
#     kmeans_model.partial_fit(distr_params.iloc[indices])

# kmeans_model.cluster_centers_.shape


# In[336]:


dist = kmeans_model.transform(distr_params)
predicted_labels = kmeans_model.predict(distr_params)
wtf = gaussian(normalize(dist, norm="l1", axis=1))

print(f"dist has shape {dist.shape}, predicted_labels has shape {predicted_labels.shape}")


# In[337]:


# wtf


# In[338]:


Counter(predicted_labels)


# In[332]:


voc_array = np.array(vocabulary)

def get_topwords2(topic):
    indices = np.where(predicted_labels == topic)[0]
    print(doc_word_distr.iloc[indices].mean(0).sort_values(ascending=False).head(10))

def get_topwords(topic):
    indices = np.where(predicted_labels == topic)[0]
    print((doc_word_distr.T * wtf[:, topic]).T.iloc[indices].mean(0).sort_values(ascending=False).head(10))

def get_topwords2(topic):
    indices = np.where(predicted_labels == topic)[0]
    xv = (doc_word_distr.T * wtf[:, topic]).T.iloc[indices]
    xvc = (xv > 0).sum(0)
    print((xv.sum(0) * calculate_trust_ratio(xvc) / xvc).sort_values(ascending=False).head(10))

def get_top2(topic):
    indices = dist[:, topic].argsort()
    print(labels[indices[:10]])
    get_topwords(topic)
    
def get_top(topic):
    indices = np.where(predicted_labels == topic)[0]
    count = Counter()
    for index in indices:
        count[labels[index]] += wtf[index, topic]
        
    print(Counter(labels[indices]))
    get_topwords(topic)


# In[333]:


get_top(0)


# In[318]:


get_top(1)


# In[309]:


get_top(2)


# In[310]:


get_top(3)


# ### LDA

# In[ ]:




