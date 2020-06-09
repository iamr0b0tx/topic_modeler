# from std lib
import re, string

# from thrid party
import numpy as np

from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups

wordnet_lemmatizer = WordNetLemmatizer()

def sigmoid(x):
    return 1 / (1 + (np.e**-x))

def gaussian(x):
    return np.e**(-x**2)

def gaussian2(x):
    return np.e**(-x)

def calculate_trust_ratio(x):
    return x / (x + 1)

def build_topic_word_distr(topics, word_topic_cos, words, topic_word_window_width, word_doc_frequency):
    topic_word_distr = pd.DataFrame(data=0.0, columns=topics, index=words)

    for topic in tqdm(range(len(topics))):
        word_topic_co = word_topic_cos[topic]
        word_word_co = pd.DataFrame(data=0.0, columns=word_topic_co[:topic_word_window_width].index, index=words)

        for index, (top_word, corelation) in enumerate(word_topic_co.items()):
            if index == topic_word_window_width:
                break

            word_word_frequency = corelation * word_doc_freqency[word_doc_freqency[top_word] > 0].sum(0)
            trust_factor = sigmoid((word_doc_freqency[top_word] > 0).sum(0))

            word_word_co[top_word] = (word_word_frequency * trust_factor) / word_doc_frequency
        topic_word_distr[topics[topic]] = word_word_co.max(1)
    return topic_word_distr

def infer_topic(label_classes, doc_vector, topic_word_distr):
    doc_topic_word_distr = topic_word_distr.copy()

    for label_class in label_classes:
        doc_topic_word_distr[label_class] *= doc_vector
    
    
    doc_topic = np.max(doc_topic_word_distr).idxmax()
    return doc_topic_word_distr, doc_topic

def get_wordnet_pos(word, use_pos):
    if not use_pos:
        return 'n'

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N":wordnet.NOUN, "V":wordnet.VERB, "r":wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# clean out the new line characters from text in docs
def clean_doc(doc, use_pos=False):
    ''' remove unwanter characters line new line '''

    unwanted_chrs = list(string.punctuation)
    # unwanted_chrs = [')', '(', '{', '}', '\t', '\n', '\r', "'", '"', "!", ",", ".", "?", ">", "<", "[", "]"]

    doc = doc.lower()
    for unwanted_chr in unwanted_chrs:
        doc = doc.replace(unwanted_chr, ' ')

    doc = word_tokenize(doc)

    word_count = len(doc)
    doc = " ".join([wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word, use_pos)) for word in doc])

    status = (len(doc) != 0 and not doc.isspace())

    return status, doc, word_count

def show_topwords(word_network, word=None, n=10, words=None):
    # if words to compared with is specified instead of topword size
    if words is not None:
        n = None

    co_occurence_ratio = word_network.get_co_occurence(word, words)
    print(f"word_node is [{word.upper()}]\n" + "="*30)

    for word, ratio in co_occurence_ratio.most_common(n):
        print(f"{word:16s} {ratio:.4f}")
    
    print()
    return

def load_data(datasize=100):
    # retrieve dataset
    docs = fetch_20newsgroups(subset='train', shuffle=False, remove=('headers', 'footers', 'quotes'))
    docs, old_labels, classes = docs.data, docs.target, docs.target_names
    
    #labels = []
    clean_docs = []

    for index, doc in enumerate(docs):
        if len(clean_docs) == datasize:
            break
            
        cd = clean_doc(doc)
        
        if len(cd) and not cd.isspace():
            clean_docs.append(cd)

    return clean_docs
