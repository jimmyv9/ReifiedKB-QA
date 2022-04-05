import sys
import string
import numpy as np
import time
from functools import partial
import gensim.downloader as api
from gensim.models import KeyedVectors
import multiprocessing.dummy as mp

from torch.utils.data import Dataset

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

def parallel_qa(query, model, entities):
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
    with open(train_path, 'r') as fin:
        all_lines = fin.readlines()
    print(len(all_lines))
    start = time.time()
    p = mp.Pool(8)
    partial_qa = partial(parallel_qa, model=model, entities=X)
    results = p.map(partial_qa, all_lines[:1000])
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

class MetaQADataset(Dataset):
    """
    Define the dataset for MetaQA
    """
    def __init__(self, data):
        """Initialize the dataset
        Args:
            data(list): a 2-D list with the structure
                        [[input, label], [], ...]
                        in details:
                        [[[entity_embedding, question_embedding, n_hop], objective index],
                         [], ...]
        """
        self.data = data
        self.length = len(data)

    def __getitem__(self, index):
        """Get data based on the index from 1 to the length of the data
        Args:
            index(int): the index number

        Return:
            data(list): one line of data with the structure [intput, label]
                        in details:
                        [[entity_embedding, question_embedding, n_hop], objective index]
        """
        return self.data[index]

    def __len__(self):
        """Get the number of questions in MetaQA
        Return:
            length(int): the number of questions in MetaQA
        """
        return self.length

if __name__ == '__main__':
    main()
