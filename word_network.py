# from std lib
from collections import Counter, defaultdict

# from third party
from tqdm import tqdm

# from lib code
from utils import calculate_trust_factor

FREQUENCY = "__frequency__"

class WordNetwork:
    def __init__(self, topics):
        self.word_nodes = {}
        self.topic_nodes = {}

        self.topics = topics

    def forward_pass(self, index, other_indices, nodes):
        node = nodes.get(index, None)

        if node is None:
            node = nodes[index] = {FREQUENCY:0}

        node[FREQUENCY] += 1

        for other_index in other_indices:
            other_index_node = node.get(other_index, None)

            if other_index_node is None:
                node[other_index] = 0

            node[other_index] += 1

        return node

    def pass_word(self, word, other_words):
        return self.forward_pass(word, other_words, self.word_nodes)

    def pass_topic(self, topic, other_words):
        return self.forward_pass(topic, other_words, self.topic_nodes)

    def get_co_occurence(self, word=None, topic=None):
        co_occurence_ratio = Counter()
        
        if word is not None:
            node = self.word_nodes.get(word, None)

        if topic is not None:
            node = self.topic_nodes.get(topic, None)

        if node is None:
            return co_occurence_ratio

        for other_index in node:
            if other_index in [FREQUENCY]:
                continue
            
            word_node = self.word_nodes.get(other_index, None)
            freq = 0 if word_node is None else word_node[FREQUENCY]

            # trust_factor = 1
            trust_factor = calculate_trust_factor(freq)
            co_occurence_ratio[other_index] = (node[other_index] * trust_factor) / freq if freq else 0

        return co_occurence_ratio

    def train(self, clean_docs, labels):
        for doc_index in tqdm(range(len(clean_docs))):
            clean_doc = clean_docs[doc_index]
            doc_words = list(set(clean_doc.split()))

            topic = labels[doc_index]
            topic_node = self.pass_topic(topic, doc_words)

            for word in doc_words:
                word_node = self.pass_word(word, doc_words)
        
        return

    def infer_topic(self, docs, labels=None):
        score = accuracy = 0
        num_of_docs = len(docs)

        for doc_index in tqdm(range(num_of_docs)):
            doc = docs[doc_index]
            doc_words = list(set(doc.split()))

            topics = Counter({label_class:0 for label_class in self.topics})

            for topic in topics:
                topic_co_occurence_ratio = self.get_co_occurence(topic=topic)
                
                for word in doc_words:
                    
                    confidence = topic_co_occurence_ratio.get(word, 0)
                    if confidence > topics[topic]:
                        topics[topic] = confidence

            if labels is not None:
                score += int(topics.most_common(1)[0][0] == labels[doc_index])
        
        if num_of_docs:
            accuracy = score / num_of_docs

        return accuracy