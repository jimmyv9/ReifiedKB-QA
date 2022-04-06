import sys
import os
import string
import numpy as np
import time
from functools import partial
import gensim.downloader as api
from gensim.models import KeyedVectors
#import multiprocessing.dummy as mp

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
            triple = triple.replace("!", "").replace(",", "")
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
    question, answer = query.split("\t")
    all_words, subj = parse_entities(question)
    # Get question embeddings
    embeddings = []
    for word in all_words:
        if word == subj:
            subj_emb = get_emb_from_model(word, model)
            embeddings.append(subj_emb)
        else:
            embeddings.append(get_emb_from_model(word, model))

    embeddings = np.array(embeddings)
    try:
        q = np.mean(embeddings, axis=0) # mean pooling
    except:
        print("Question {} could not be parsed".format(question))
    # Get answer as mapped entities
    answer.replace(",","").replace("!", "")
    all_answers = answer.strip().split('|')
    ans_ids = []
    for ans in all_answers:
        ans_ids.append(entities[ans])
    ans_ids = np.array(ans_ids)
    return (q, subj_emb, ans_ids)

def get_emb_from_model(word, model):
    if word in model.index_to_key:
        return model[word]
    elif word.lower() in model.index_to_key:
        return model[word.lower()]
    else:
        return model["UNK"]

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
    sentence = sentence.replace(",", "").replace("!", "")
    for word in sentence.split():
        if word[0] == "[" and word[-1] == "]":
            entity = word[1:-1]
            tokens.append(word[1:-1])
        elif entity_found == True:
            if word[-1] == ']':
                entity += "_" + string.capwords(word[:-1])
                tokens.append(entity)
                entity_found = False
            else:
                entity += "_" + string.capwords(word)
        elif word[0] == '[':
            entity_found = True
            entity = string.capwords(word[1:])
        else:
            tokens.append(word)
    return tokens, entity

def get_pytorch_input(input_dir, nhop):
    files = os.listdir(input_dir)
    train_data = []
    for f in files:
        path = input_dir + "/" + f
        with open(path, 'r') as fin:
            for line in fin:
                line = line.split('\t')
                q = line[0].strip()
                q = np.array([float(x) for x in q.split()])
                subj = line[1].strip()
                subj = np.array([float(x) for x in subj.split()])
                a = line[2].strip().split()
                a = [int(x) for x in a]
                for ent in a:
                    instance = [[q, subj, nhop], ent]
                    train_data.append(instance)
    return train_data


def main():
    if len(sys.argv) != 7:
        print("Execute `python read_metaQA.py path_to_kb path_to_trainQA path_to_pretrained_embeddings outfile a b`")
        sys.exit(-1)
    kb_path = sys.argv[1]
    train_path = sys.argv[2]
    emb_path = sys.argv[3]
    outfile_path = sys.argv[4]
    a = int(sys.argv[5])
    b = int(sys.argv[6])

    triples = read_triples(kb_path)
    X = extract_entities(triples)
    R = {t[1] for t in triples} # extract relations
    start = time.time()
    model = api.load(emb_path)
    with open(train_path, 'r') as fin:
        all_lines = fin.readlines()
    print("Loading gensim model took {} seconds".format(time.time()-start))
    partial_qa = partial(parallel_qa, model=model, entities=X)
    start = time.time()
    results = map(partial_qa, all_lines[a:b])
    a = -1
    b = -1
    print("Execution took {} seconds".format(time.time()-start))
    #training_vals = read_qa(train_path, model, X)
    start = time.time()
    with open(outfile_path, 'w') as fout:
        for qa in results:
            q = qa[0]
            subj = qa[1]
            a = qa[2]
            to_print = str(q).replace("\n", "")[1:-1] + "\t" # add question
            to_print += str(subj).replace("\n", "")[1:-1] + "\t" # add subject's embedding
            to_print += str(a).replace("\n", "")[1:-1] + "\n" # add answer
            fout.write(to_print)
    print("Writing {} took {} seconds".format(outfile_path, time.time()-start))
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
