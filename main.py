# from third party lib
from sklearn.model_selection import train_test_split

# from lib code
from word_network import WordNetwork
from utils import load_data, show_topwords

def main():
    # fetch data
    docs, labels, label_classes = load_data(100)
    print(f"there are {len(docs)} docs and {len(label_classes)} classes: {label_classes}")
    
    # split data to dev and val
    train_docs, test_docs, train_labels, test_labels = train_test_split(docs, labels, test_size=.33, random_state=42, shuffle=False)
    
    # initialize network
    word_network = WordNetwork(label_classes)

    # train the network
    print("\nTraining\n" + "="*30)
    word_network.train(train_docs, train_labels)
    print(f"discovered {len(word_network.word_nodes)} words and {len(word_network.topic_nodes)} involved!")  

    # # show topwords related to computer
    # show_topwords(word_network, word="computer")
    
    # # show topwords related to a topic
    # for topic in label_classes:
    #     show_topwords(word_network, topic=topic)

    print("\nBuilding Recursive Topic Model\n" + "="*30)
    word_network.build_topic_model()

    print("\nEvaluation\n" + "="*30)
    test_accuracy, test_accuracy1 = word_network.infer_topic(test_docs, test_labels)
    print(f"test_accuracy {test_accuracy*100:.2f}%, recursive => {test_accuracy1*100:.2f}%")

if __name__ == "__main__":
    main()