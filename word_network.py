import numpy as np
from scipy.stats import entropy as calculate_entropy

from collections import Counter

from node import Node
from utils import sigmoid

class WordNetwork:
    def __init__(self, threshold=1):
        # all word_nodes connected
        self.word_nodes = {}

        # the entropy threshold
        self.threshold = threshold

    def pass_word(self, word, doc, update=False):
        # fetch word_node
        word_node = self.word_nodes.get(word, None)

        # id node does not exist, create one
        if word_node is None:
            word_node = self.word_nodes[word] = Node(word)

        # pass doc t word_node
        doc_size = word_node.pass_doc(doc)

        if update:
            self.update_relations(word_node)

        return word_node

    def update_relations(self, primary_word_node):
        co_occurance_ratios = []
        word_node_co_occurence_ratios = Counter()

        for secondary_word, secondary_word_node in self.word_nodes.items():
            if primary_word_node.word == secondary_word:
                continue
            
            # the appearance of primary_word_node and secondary_word_node to gether in a doc
            co_occurance_ratio = len(primary_word_node.docs.intersection(secondary_word_node.docs)) / len(primary_word_node.docs)
            co_occurance_ratios.append(co_occurance_ratio)

            if co_occurance_ratio:
                word_node_co_occurence_ratios[secondary_word] = co_occurance_ratio
        
        entropy = calculate_entropy(co_occurance_ratios, base=2)
        print(entropy, co_occurance_ratios)
        if entropy > self.threshold:
            return

        primary_word_node.entropy = entropy# * (1 - sigmoid(len(primary_word_node.docs)))
        primary_word_node.word_node_co_occurence_ratios = word_node_co_occurence_ratios
