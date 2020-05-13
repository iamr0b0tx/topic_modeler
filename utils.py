# from std lib
import re, string

# from thrid party
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def calculate_trust_factor(x):
    return 1 / (1 + (np.e**-x))
    # return x / (x + 1)

# clean out the new line characters from text in docs
def clean_document(doc):
    ''' remove unwanter characters line new line '''

    unwanted_chrs = [')', '(', '{', '}', '\t', '\n', '\r', "'", '"', "!", ".", ":", "-", ".", ","]
    # unwanted_chrs = string.punctuation
    
    doc = doc.lower()
    for unwanted_chr in unwanted_chrs:
        # doc = re.sub(f"{unwanted_chr}+", " ", doc)
        doc = doc.replace(unwanted_chr, ' ')

    return doc.strip()

def show_topwords(word_network, word=None, topic=None, n=10):
    co_occurence_ratio = word_network.get_co_occurence(word=word, topic=topic)

    title, title_ = ("topic", topic) if topic is not None else ("word", word)
    print(f"{title}_node is [{title_.upper()}]\n" + "="*30)

    for word, ratio in co_occurence_ratio.most_common(n):
        print(f"{word:16s} {ratio:.4f}")
    
    print()
    return

def load_data(datasize=100):
    # retrieve dataset
    docs = fetch_20newsgroups(subset='train', shuffle=False, remove=('headers', 'footers', 'quotes'))
    docs, old_labels, classes = docs.data[:datasize], docs.target[:datasize], docs.target_names
    
    # the new classes
    labels = []
    label_classes = list(set([x.split('.')[0] for x in classes]))

    # restructuring classes  from 19 to less
    for label in old_labels:
        labels.append(classes[label].split('.')[0])

    # clean the documents
    clean_docs = [clean_document(doc) for doc in docs]

    return clean_docs, labels, label_classes
