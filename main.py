# from lib code
from word_network import WordNetwork
from utils import load_data, show_topwords

def main():
    # fetch data
    docs = load_data(10000)
    print(f"there are {len(docs)} docs")
    
    # initialize network
    word_network = WordNetwork()

    # train the network
    print("\nTraining\n" + "="*30)
    word_network.train(docs)
    print(f"discovered {len(word_network.word_nodes)} words!\n")  

    # show topwords related to computer
    show_topwords(word_network, "computer")
    show_topwords(word_network, "88")
    show_topwords(word_network, "typist")

if __name__ == "__main__":
    main()