import sys
import string
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

def read_qa(filename, model, entities):
    """
    Extracts question answer as question answer pairs, where question is
    represented as an embedding and answers are represented by their entity
    index

    Input:
        filename: path to test_qa text files
        model: loaded pre-trained model with embeddings

    Output:
        QA: ([q_emb],x_i), where q_emb is the embedding of words in question i
        and x_i is the entity index for the respective question
    """

    with open(filename, 'r') as fin:
        testing_instances = []
        i = 0
        for line in fin:
            if i == 20:
                break
            i += 1
            question, answer = line.split("\t")
            all_words = parse_entities(question)
            # Get question embeddings
            embeddings = []
            for word in all_words:
                if word in model.index_to_key:
                    embeddings.append(model[word])
                elif word.lower() in model.index_to_key:
                    embeddings.append(model[word.lower()])
                else:
                    embeddings.append(model["UNK"])
            embeddings = np.array(embeddings)
            print("Tokens: {} yield embedding of length {}".format(all_words, embeddings.shape))
            try:
                q = np.mean(embeddings, axis=0) # mean pooling
            except:
                print("Question {} could not be parsed".format(question))
            # Get answer as mapped entities
            all_answers = answer.strip().split('|')
            for ans in all_answers:
                testing_instances.append((q, entities[ans]))
    return testing_instances

def parse_entities(sentence):
    """
    Tokenizes sentence keeping entities together
    """
    tokens = []
    entity_found = False
    entity = ""
    for word in sentence.split():
        if word[0] == "[" and word[-1] == "]":
            tokens.append(word[1:-1])
        elif entity_found == True:
            if word[-1] == ']':
                entity += "_" + string.capwords(word[:-1])
                tokens.append(entity)
                entity_found = False
                entity = ""
            else:
                entity += "_" + string.capwords(word)
        elif word[0] == '[':
            entity_found = True
            entity = string.capwords(word[1:])
        else:
            tokens.append(word)
    return tokens

def main():
    if len(sys.argv) != 4:
        print("Execute `python read_metaQA.py path_to_kb path_to_trainQA path_to_pretrained_embeddings`")
        sys.exit(-1)
    kb_path = sys.argv[1]
    train_path = sys.argv[2]
    emb_path = sys.argv[3]

    triples = read_triples(kb_path)
    X = extract_entities(triples)
    R = {t[1] for t in triples} # extract relations
    print(R)
    model = api.load(emb_path)
    training_vals = read_qa(train_path, model, X)
    with open("./to_train.txt", 'w') as fout:
        for t in training_vals:
            to_print = ""
            to_print += str(list(t[0][1:-1])) + "\t" + str(t[1]) + "\n"
            fout.write(to_print)
    return 0



if __name__ == '__main__':
    main()
