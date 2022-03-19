import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

def read_triples(filename):
    """
    Reads triples from MetaQA knowledge base

    Input:
        filename (str): relative path to MetaQA file

    Output:
        triples (set): a set of triples which represent two nodes
        subject and object on a KG, related by a relation edge. Each tuple is
        formatted in the set as (subj, rel, obj)
    """
    triples = set()
    with open(filename, 'r') as fin:
        for triple in fin:
            triple = triple.split("|")
            triple[2] = triple[2].strip("\n")
            triple = tuple(triple)
            triples.add(triple)
    return triples

def extract_entities(triples):
    """
    Extracts entities subject and object, representative of nodes in KG,
    mapping each of these with a unique index from 1 to N.

    Input:
        triples set((subj, rel, obj)): set of triples from KB
    Output:
        X (dict): dictionary mapping N entities to a unique index. Each entity
        x -> i, where i < N
    """
    entities = set()
    for t in triples:
        entities.add(t[0])
        entities.add(t[2])
    X = dict()
    i = 1
    for e in entities:
        X[e] = i
        i += 1
    return X



def main():
    triples = read_triples("MetaQA/kb.txt")
    X = extract_entities(triples)
    print(len(X))
    R = {t[1] for t in triples} # extract relations
    print(len(R))
    print(R)
    model = api.load("word2vec-google-news-300")
    model.most_similar("glass")
    return 0



if __name__ == '__main__':
    main()
