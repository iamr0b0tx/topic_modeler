from scipy.stats import entropy
from sklearn.datasets import fetch_20newsgroups

from word_network import WordNetwork

def main():
    wn = WordNetwork()

    # total number of samples needed
    datasize = 10

    # retrieve dataset
    docs = fetch_20newsgroups(subset='train', shuffle=False, remove=('headers', 'footers', 'quotes'))
    docs, labels, classes = docs.data[:datasize], docs.target[:datasize], docs.target_names

    # the actual labels as np array
    labels = np.array(labels)

    # the new classes
    label_classes = list(set([x.split('.')[0] for x in classes]))

    # restructuring classes  from 19 to less
    for label, cl in enumerate(classes):
        labels[labels == label] = label_classes.index(cl.split('.')[0])

    print(f"there are {len(docs)} docs and {len(label_classes)} classes")

    # clean words out
    clean_docs = clean_text(docs)

    # get each doc
    for clean_doc in clean_docs:

        # tokenize words
        doc_words = clean_doc.split(" ")

        for word in doc_words:
            wn.pass_word(word)

    print(f"WordNetwork has {len(wn.word_nodes)} word_nodes created!")

if __name__ == "__main__":
    main()