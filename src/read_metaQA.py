import sys
import string
import numpy as np
import time
from functools import partial
import gensim.downloader as api
from gensim.models import KeyedVectors
import multiprocessing.dummy as mp

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

def extract_entity_embeddings(entities, model):
    """
    Extracts the embeddings from our model, and maps each entity to its unique
    embedding
    """
    entity_embs = dict()
    for ent in entities:
        ent = ent.split()
        ent = "_".join(ent)
        if ent in model.index_to_key:
            entity_embs[ent] = model[word]
        elif ent.lower() in model.index_to_key:
            entity_embs[ent] = model[word.lower()]
        else:
            entity_embs[ent] = model["UNK"]
    return entity_embs

def write_entity_embeddings(entity_embs, infile, outfile):
    """
    Outputs a file with entity embeddings for each entity subject in the order
    they are found in the entity answer
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            for line in fin:
                sent = line.strip().split("[")
                entity = sent[1].split("]")[0]
                entity = "_".join(entity.split())
                fout.write(str(entity_embs[entity])[1:-1])


def parallel_qa(query, model, entities):
    """
    Takes in a single question, answer pair and returns the essential
    components to train our neural networks

    Inputs:
        query (str): line of training file containing question and answer
        model: gensim pretrained loaded model containing word embeddings
        entities (dict): set of entities, mapped to their unique integer ID

    Outputs:
        testing_instances (q, A): q is the mean-pooled array which averages
        each dimension of the embeddings in the question, A is the ID of the
        entity in the answer
    """
    testing_instances = []
    question, answer = query.split("\t")
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
    try:
        q = np.mean(embeddings, axis=0) # mean pooling
    except:
        print("Question {} could not be parsed".format(question))
    # Get answer as mapped entities
    all_answers = answer.strip().split('|')
    for ans in all_answers:
        testing_instances.append((q, entities[ans]))
    return testing_instances



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
        for line in fin:
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
    model = api.load(emb_path)
    X_embs = extract_entity_embeddings(X, model)
    write_entity_embeddings(X_embs, train_path, "subj_embeddings.txt")
    with open(train_path, 'r') as fin:
        all_lines = fin.readlines()
    print(len(all_lines))
    start = time.time()
    p = mp.Pool(8)
    partial_qa = partial(parallel_qa, model=model, entities=X)
    results = p.map(partial_qa, all_lines)
    p.close()
    p.join()
    print("Execution took {} seconds".format(time.time()-start))

    #training_vals = read_qa(train_path, model, X)
    with open("./to_train.txt", 'w') as fout:
        for question in results:
            for ans in question:
                to_print = ""
                to_print += str(list(ans[0][1:-1])) + "\t" + str(ans[1]) + "\n"
                fout.write(to_print)
    return 0



if __name__ == '__main__':
    main()
