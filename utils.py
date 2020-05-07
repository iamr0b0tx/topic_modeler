import numpy as np

def clean_text(doc):
    ''' remove unwanter characters line new line '''
    
    unwanted_chrs = [')', '(', '{', '}', '\t', '\n', '\r', "'", '"', "!"]
    for unwanted_chr in unwanted_chrs:
        doc = doc.replace(unwanted_chr, ' ')
    
    return doc.strip()

def sigmoid(x):
    return 1 / (1 + (np.e**-x))

def word_tokenize(docs):
    if type(docs) == str:
        docs = [docs]

    # get each doc
    for doc_index, doc in enumerate(docs):
        
        # clean and tokenize words
        doc_words = clean_text(doc).split(" ")
        
        for doc_word in doc_words:
            if doc_word:
                yield doc_index, doc_word