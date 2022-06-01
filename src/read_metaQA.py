import sys
import os
import re
import string
import numpy as np
import yaml
from functools import partial
import gensim.downloader as api
from gensim.models import KeyedVectors, Word2Vec

import torch
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
    i = 0
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
        if ent in model.key_to_index:
            entity_embs[ent] = model[word]
        elif ent.lower() in model.key_to_index:
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
    answer = answer.replace(",","").replace("!", "")
    all_answers = answer.strip().split('|')
    ans_ids = []
    for ans in all_answers:
        if ans in entities:
            ans_ids.append(entities[ans])
    ans_ids = np.array(ans_ids)
    question = question.replace("!", "").replace(",", "")
    new_subj = question.split("[")[1]
    new_subj = new_subj.split("]")[0]
    return (entities[new_subj], q, ans_ids)

def get_emb_from_model(word, model):
    if word in model.key_to_index:
        return model[word]
    elif word.lower() in model.key_to_index:
        return model[word.lower()]
    return np.zeros(len(model[0]))

def parse_entities(sentence):
    """
    Tokenizes sentence keeping entities together
    """
    entity_found = False
    sentence, entity = parse_question(sentence)
    tokens = sentence.split()
    return tokens, entity

def parse_question(question):
    question = question.replace(",", "").replace("!", "")
    match = re.search("\[(.*)\]", question)
    entity = match.group()
    new_entity = "_".join(entity[1:-1].split())
    question = question.replace(entity, new_entity)
    return question, new_entity

def main(config):
    if len(sys.argv) != 4:
        # model_name = {meta_qa, wqsp}
        print("Execute `python read_metaQA.py path_to_kb path_to_pretrained_embeddings model_name`")
        sys.exit(-1)
    kb_path = sys.argv[1]
    emb_path = sys.argv[2]
    name = sys.argv[3]

    ent_path = config['entity_path_'+name]
    rel_path = config['relation_path_'+name]
    triples = read_triples(kb_path)
    X = extract_entities(triples)
    model = Word2Vec.load(emb_path)
    model = model.wv
    print("W2V model loaded")

    with open(ent_path, 'w') as fout:
        for ent in X:
            fout.write("{}\t{}\n".format(ent, X[ent]))
    print("Finished writing entity map")

    if name == 'meta_qa':
        R = {}
        for triple in triples:
            r = triple[1]
            if r not in R:
                R[r] = len(R)
                r_rev = "_".join(["rev", r])
                R[r_rev] = len(R)

        with open(rel_path, 'w') as fout:
            for rel in R:
                fout.write("{}\t{}\n".format(rel, R[rel]))

        # Generate embedded questions
        hops = ['1_hop', '2_hop', '3_hop']
        files = ['qa_train.txt', 'qa_dev.txt', 'qa_test.txt']
        for hop in hops:
            for infile in files:
                with open(config[hop+"_path"]+"/"+infile, 'r') as fin:
                    all_lines = fin.readlines()
                partial_qa = partial(parallel_qa, model=model, entities=X)
                start = time.time()
                results = map(partial_qa, all_lines)
                file_size = 0
                with open(config[hop+"_output"]+infile, 'w') as fout:
                    for qa in results:
                        file_size += 1
                        if file_size % 1000 == 0:
                            print("Processed {} questions".format(file_size))
                        subj = qa[0]
                        q = qa[1]
                        a = qa[2]
                        to_print = str(subj) + "\t" # add subject's embedding
                        to_print += str(q).replace("\n", "")[1:-1] + "\t" # add question
                        to_print += str(a).replace("\n", "")[1:-1] + "\n" # add answer
                        fout.write(to_print)

    elif name == 'wqsp':
        R = {'1':{}, '2':{}, '3':{}}
        for triple in triples:
            r = triple[1]
            r_type = r.split('_')[-1]
            if r not in R:
                R[r_type][r] = len(R[r_type])
                #r_rev = "_".join(["rev", r])
                #R[r_rev] = len(R)

        for t in R:
            relation_file = config['rel{}_wqsp'.format(t)]
            with open(relation_file, 'w') as fout:
                for rel in R[t]:
                    fout.write("{}\t{}\n".format(rel, R[t][rel]))

        size_2 = len(R['2'])
        size_3 = len(R['3'])
        for key in R['2']:
            R['2'][key] += size_2
        for key in R['3']:
            R['3'][key] += size_2 + size_3

        with open(config['relation_path_wqsp'], 'w') as fout:
            for t in R:
                for rel in R[t]:
                    fout.write("{}\t{}\n".format(rel, R[t][rel]))
        print("Finished writing WQSP relation maps")
        # Generate question embeddings
        files = ['train', 'dev', 'test']
        for mode in files:
            with open(config[mode+'_path_proc'], 'r') as fin:
                all_lines = fin.readlines()
            partial_qa = partial(parallel_qa, model=model, entities=X)
            results = map(partial_qa, all_lines)
            with open(config['embedding_wqsp']+'/'+mode+'.txt', 'w') as fout:
                for qa in results:
                    if len(qa[2]) == 0:
                        # Some values, 2 in train and 3 in test could not be
                        # found in Freebase, so we simply "skip" them
                        continue
                    subj = qa[0]
                    q = qa[1]
                    a = qa[2]
                    to_print = str(subj) + "\t" # add subject's embedding
                    to_print += str(q).replace("\n", "")[1:-1] + "\t" # add question
                    to_print += str(a).replace("\n", "")[1:-1] + "\n" # add answer
                    fout.write(to_print)
    return 0

def read_metaqa(input_file):
    train_data = []
    with open(input_file, 'r') as fin:
        # for debugging
        #cnt = 0
        for idx, line in enumerate(fin):
            line = line.split('\t')
            subj = int(line[0].strip())
            q = line[1].strip()
            q = torch.tensor([float(x) for x in q.split()]).unsqueeze(0)
            a = line[2].strip().split()
            a = [int(x) for x in a]
            instance = [idx, subj, q, a] # refer to page 6, paragraph 3, 2nd sentence
            train_data.append(instance)
            # for debugging
            #cnt += 1
            #if cnt >= 2:
            #    return train_data
    return train_data

def read_KB(config):
    """Read knowledge graph from file and convert to COO matrixes
    Args:
        file_path(str): the path of the kb

    Returns:
        M_subj(COO_Matrix): a sparse matrix with size (# of triples, # of entities)
        M_rel(COO_Matrix): a sparse matrix with size (# of triples, # of relation)
        M_obj(COO_Matrix): a sparse matrix with size (# of triples, # of entities)
    """
    with open(config['entity_path'], 'r') as f:
        X = {}
        for line in f.readlines():
            entity, idx = line.split('\t')
            idx = int(idx)
            X[entity] = idx
        X_rev = [''] * len(X)
        for k, v in X.items():
            X_rev[v] = k

    with open(config['relation_path'], 'r') as f:
        R = {}
        for line in f.readlines():
            relation, idx = line.split('\t')
            idx = int(idx)
            R[relation] = idx

    triples = read_triples(config['kb_path'])
    total_triples = len(triples)
    subj_idx = []
    rel_idx = []
    obj_idx = []
    for idx, triple in enumerate(triples):
        s, r, o = triple
        subj_idx.append([idx, X[s]])
        rel_idx.append([idx, R[r]])
        obj_idx.append([idx, X[o]])
        # add a reverse edge
        r_rev = '_'.join([r, 'rev'])
        subj_idx.append([idx + total_triples, X[o]])
        rel_idx.append([idx + total_triples, R[r_rev]])
        obj_idx.append([idx + total_triples, X[s]])
    subj_data = torch.FloatTensor([1] * len(subj_idx))
    subj_idx = torch.LongTensor(subj_idx).T
    subj_size = [len(triples) * 2, len(X)] # add reverse edges
    rel_data = torch.FloatTensor([1] * len(rel_idx))
    rel_idx = torch.LongTensor(rel_idx).T
    rel_size = [len(triples) * 2, len(R)] # add reverse edges
    obj_data = torch.FloatTensor([1] * len(obj_idx))
    obj_idx = torch.LongTensor(obj_idx).T
    obj_size = [len(triples) * 2, len(X)] # add reverse edges

    M_subj = torch.sparse.FloatTensor(subj_idx, subj_data, torch.Size(subj_size))
    M_rel = torch.sparse.FloatTensor(rel_idx, rel_data, torch.Size(rel_size))
    M_obj = torch.sparse.FloatTensor(obj_idx, obj_data, torch.Size(obj_size))

    #return M_subj, M_rel, M_obj
    return M_subj, M_rel, M_obj, X_rev # for debugging

class MetaQADataset(Dataset):
    """
    Define the dataset for MetaQA
    """
    def __init__(self, data, no_of_entities):
        """Initialize the dataset
        Args:
            data(list): a 2-D list with the structure in details:
                        [[question index, entity_index, question_embedding, objective indexes],
                         [], ...]
            no_of_entities(int): the total number of entities in KB
        """
        self.data = data
        self.length = len(data)
        self.no_of_entities = no_of_entities

    def __getitem__(self, index):
        """Get data based on the index from 1 to the length of the data
        Args:
            index(int): the index number

        Return:
            data(list): one line of data with the structure
                        [intput, label], total_number_of_entities
                        in details:
                        [question index, entity_index, question_embedding, objective indexes],
                        total_number_of_entities
        """
        return self.data[index], self.no_of_entities

    def __len__(self):
        """Get the number of questions in MetaQA
        Return:
            length(int): the number of questions in MetaQA
        """
        return self.length

if __name__ == '__main__':
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    main(config)
