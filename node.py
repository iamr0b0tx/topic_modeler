from collections import Counter

class Node:
    def __init__(self, word):
        # the word itself
        self.word = word

        # docs that word exists in
        self.docs = set()

        self.entropy = 1000
        self.word_node_co_occurence = Counter()

    def pass_doc(self, doc):

        self.docs.add(doc)
        return len(self.docs)
