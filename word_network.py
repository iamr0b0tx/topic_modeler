# from std lib
from collections import Counter, defaultdict

# from third party
from tqdm import tqdm

# from lib code
from utils import sigmoid, gaussian

FREQUENCY = "__frequency__"

class WordNetwork:
    def __init__(self):
        self.word_nodes = {}

    def pass_word(self, word, doc_vector):
        word_node = self.word_nodes.get(word, None)

        if word_node is None:
            word_node = self.word_nodes[word] = {FREQUENCY:0}

        word_node[FREQUENCY] += 1

        for other_word, other_word_rf in doc_vector.items():
            other_word_node = word_node.get(other_word, None)

            if other_word_node is None:
                word_node[other_word] = 0

            word_node[other_word] +=  1

        return

    def get_co_occurence(self, word, words=None):
        co_occurence_ratio = Counter()
        word_node = self.word_nodes.get(word, None)

        if word_node is None:
            return co_occurence_ratio

        if words is None:
            words = list(word_node.keys())

        for other_word in words:
            if other_word in [FREQUENCY]:
                continue
            
            other_word_node = self.word_nodes.get(other_word, None)
            if other_word_node is None:
                co_occurence_ratio[other_word] = 0
                continue

            freq = other_word_node[FREQUENCY]
            trust_factor = sigmoid(freq)

            co_occurence_ratio[other_word] = (word_node.get(other_word, 0) * trust_factor) / freq if freq else 0
        return co_occurence_ratio

    def train(self, clean_docs):
        for doc_index in tqdm(range(len(clean_docs))):
            clean_doc = clean_docs[doc_index]

            doc_words = clean_doc.split()
            doc_word_freq =  Counter(doc_words)

            # holds words and relative frequency in document
            doc_vector = {key:value/len(doc_words) for key, value in doc_word_freq.items()}

            for word in doc_vector:
                self.pass_word(word, doc_vector)
        
        return