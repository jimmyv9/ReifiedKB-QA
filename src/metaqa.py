import re
import os
import math
import logging
import multiprocessing
from functools import partial

import yaml
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaModel
from gensim.models import Word2Vec

def get_logger(file_name=None):
    """Set log for the whole file
    Args:
        file_name(str): the file we want to record the log
    """
    logger = logging.getLogger('train')  # set logger's name
    logger.setLevel(logging.INFO)  # set logger's level
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()  # console handler
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if file_name is not None and '' != file_name:
        fh = logging.FileHandler(file_name, mode='w')  # file handler, mode rewrite
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class MetaQADataset(Dataset):
    """ Define the dataset for MetaQA

    Attributes:
        data(list): a 2-D list with the structure in details:
                    [[question index, entity_index, question_embedding,
                      objective indexes], [], ...]
        no_of_entities: number of entities in the dataset
    """
    def __init__(self, data, no_of_entities):
        self.data = None
        self.no_of_entities = 0

    def __getitem__(self, index):
        """Get data based on the index from 0 to the length of the data - 1

        Args:
            index(int): the index number

        Return:
            data(list): one line of data with the structure
                        [intput, label], total_number_of_entities
                        in details:
                        [question index, entity_index, question_embedding,
                         objective indexes], total_number_of_entities
        """
        return self.data[index], self.no_of_entities

    def __len__(self):
        """Get the number of questions in MetaQA
        Return:
            length(int): the number of questions in MetaQA
        """
        if self.data is None:
            return 0
        return len(data)

class MetaQAReader():
    """ Preprocess, read, and write MetaQA

    Attributes:
        triples(set):
            a set of triples which represent two nodes subject and object
            on a KG, related by a relation edge. Each tuple is formatted
            in a tuple as (subj, rel, obj)

        entities(dict):
            map entities to index. Its structure is {entity:index}

        relations(dict):
            map relations to index. Its structure is {relation:index}

        logger(logging):
            log everything in the class
    """
    def __init__(self, config):
        self.config = config
        self.triples = None # knowledge base triples
        self.entities = None # all entities and their index
        self.relations = None # all relations and their index

        # create logger for the reader
        log_dir = self.config['logger_dir']
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_name = 'log_MetaQA_{}.txt'.format(self.config['embedding_method'])
        self.logger = get_logger(os.path.join(config['logger_dir'], log_file_name))

    def load_kb(self):
        """ Reads triples from MetaQA knowledge base

        """
        self.logger.info('Loading knowledge base to triples')
        triples = set()
        with open(self.config['kb_path'], 'r') as fin:
            for triple in fin:
                triple = triple.strip()
                triple = triple.replace("!", "").replace(",", "")
                triple = triple.split("|")
                triple = tuple(triple)
                triples.add(triple)

        self.triples = triples

    def parse_kb(self):
        """ Parse knowledge base triples
        
        Extracts entities subject, relations, and object, representative
        of nodes in KG, mapping each of these with a unique index from 0 to N-1.
        """
        self.logger.info('Parsing triples to entities and relations')
        # gather all entities and relations
        entities = set()
        relations = set()
        for t in self.triples:
            entities.add(t[0])
            entities.add(t[2])
            relations.add(t[1])
            relations.add('_'.join([t[1], 'rev'])) # add a reverse edge

        # sort to make sure the index is identity if pre-trained
        # entity file and relation file are missed
        entities = list(sorted(entities))
        relations = list(sorted(relations))

        # assgin index and convert to dictionary
        self.entities = {e:i for i, e in enumerate(entities)}
        self.relations = {r:i for i, r in enumerate(relations)}

    def parse_question(self, question):
        """ Parse question strings to question itself and its corresponding entity

        Args:
            question(str): a question text with format "xx ... [entity] ... xx"

        Returns:
            question(str): a question text with format "xx ... xx"
            entity(str): a entity string with format "xx_..._xx"
        """
        question = question.replace(",", "").replace("!", "")
        match = re.search("\[(.*)\]", question)
        entity = match.group()
        if 'Word2vec' == self.config['embedding_method']:
            entity = "_".join(entity[1:-1].split())
            question = question.replace(entity, entity)
        else:
            entity = entity[1:-1]
        return question, entity

    def parse_query(self, file_path):
        """ Read and parse queries in the file from file_path

        Args:
            file_path(str): path of the file

        Returns:
            queries(list): a list of parsed queries with struture:
                           [[index, entity, question, answer], [...], ...]
        """
        self.logger.info('Parsing questions to question text and entities')
        with open(file_path, 'r') as f:
            lines = f.readlines()

        queries = []
        for idx, query in enumerate(lines):
            query = query.strip()
            question, answer = query.split('\t')

            # parse question and answer
            question, entity = self.parse_question(question)
            answer = answer.replace("!", "").replace(",", "")
            answer = answer.split('|')

            queries.append([idx, entity, question, answer])

        return queries

    def load_saved_query(self, file_path):
        """ Read saved data from file_path

        Args:
            file_path(str): the path of saved processed file

        Returns:
            data(list): a 2-D list with the structure in details:
                        [[question index, entity_index, question_embedding,
                          objective indexes], [], ...]
        """
        self.logger.info('Loading saved processed query file')
        data = []
        with open(file_path, 'r') as fin:
            # for debugging
            if self.config['DEBUG']:
                cnt = 0
            for line in fin:
                line = line.split('\t')
                query_id = int(line[0].strip())
                subj_id = int(line[1].strip())
                query_embedding = line[2].strip()
                query_embedding = torch.tensor(
                        [float(x) for x in query_embedding.split()]).unsqueeze(0)
                answers_id = line[3].strip().split()
                answers_id = [int(x) for x in answers_id]

                # refer to page 6, paragraph 3, 2nd sentence
                instance = [query_id, subj_id, query_embedding, answers_id]
                data.append(instance)
                # for debugging
                if self.config['DEBUG']:
                    cnt += 1
                    if cnt >= 2:
                        return data
        return data

    def parallel_word2vec(self, query, model):
        """ Takes in a single question, answer pair and returns the essential
            components to train our neural networks
    
        Args:
            query(list):
                a list of parsed queries with struture:
                [[index, entity, question, answer], [...], ...]
            model:
                gensim pretrained loaded model containing word embeddings
    
        Returns:
            query(list):
                a list includes query id, query entity id, query embedding,
                and answer ids. The structure looks like:

                (query_id, entity_id, query_embedding, answer_ids)

                where query_id and entity id are integers, query_embedding is
                the mean-pooled numpy array which averages each dimension of
                the embeddings in the question, answer_ids is a list of IDs
                of the entity in the answer format in numpy array
        """
        idx, entity, question, answer = query

        entity_id = self.entities[entity]

        # Get question embeddings
        embeddings = []
        for word in question.split():
            if word in model.key_to_index:
                emb = model[word]
            elif word.lower() in model.key_to_index:
                emb = model[word.lower()]
            else:
                emb = np.zeros(len(model[0]))
            embeddings.append(emb)
    
        embeddings = np.array(embeddings)
        try:
            query_embedding = np.mean(embeddings, axis=0) # mean pooling
        except:
            print("Question {} could not be parsed".format(question))

        # Get answer as mapped entities
        answers_ids = [self.entities[answer] for answer in answers]
        answers_ids = np.array(answers_ids)

        query = [idx, entity_id, query_embedding, answers_ids]
    
        return query

    def sequential_bert(self, queries, tokenizer, model):
        """ Map entities to ids and embed quesitions
    
        Args:
            queries(list):
                a list of parsed queries with struture:
                [[index, entity, question, answer], [...], ...]
            tokenizer:
                BERT tokenizer
            model:
                pre-trained BERT model
    
        Returns:
            processed_queries(list):
                a list of processed queries, eash element in the list represents the
                processed result of the query. Each tuple looks like:

                (query_id, entity_id, query_embedding, answer_ids)

                where query_id and entity id are integers, query_embedding is
                the mean-pooled numpy array which averages each dimension of
                the embeddings in the question, answer_ids is a list of IDs
                of the entity in the answer format in numpy array
        """
        self.logger.info('Mapping entitis to ids and embedding questions by BERT')
        no_batches = math.ceil(len(queries) / self.config['batch_size'])
        processed_queries = []
        for i in tqdm(range(no_batches), desc='Generating BERT embeddings'):
            start = i * self.config['batch_size']
            end = (i + 1) * self.config['batch_size']
            batched_queries = queries[start:end]

            idxes, entities, batched_questions, batched_answers = list(zip(*batched_queries))

            # convert entities to their corresponding index
            entities_id = [self.entities[entity] for entity in entities]

            # tokenize questions and convert to embeddings
            batched_encoded_inputs = tokenizer(batched_questions,
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt')
            for k, v in batched_encoded_inputs.items():
                batched_encoded_inputs[k] = v.cuda()
            with torch.no_grad():
                embed = model(**batched_encoded_inputs)
                queries_embedding = embed.pooler_output.cpu().detach().numpy()

            # Get answer as mapped entities
            answers_id = [np.array([self.entities[answer] for answer in answers])
                           for answers in batched_answers]

            batched_queries = list(zip(idxes, entities_id, queries_embedding, answers_id))
            processed_queries.extend(batched_queries)

        return processed_queries

    def load_query(self):
        """ Read from saved data or generate new datasets

        Returns:
            data(dict):
                include data for train, dev, and test part with structure:
                {'train': data, 'dev': data, 'test: data}

                for each data, its description is as follows:
                data(list): a 2-D list with the structure in details:
                            [[question index, entity_index, question_embedding,
                              objective indexes], [], ...]
        """
        n_hop = self.config['n_hop']
        data_sets = ['train', 'dev', 'test']
        data = {}

        # if file exists, read and return
        if ('processed_query_dir' in self.config and
                self.config['processed_query_dir'] is not None):
            query_dir = os.path.join(self.config['processed_query_dir'],
                                     '{}-hop'.format(n_hop))
            if os.path.exists(query_dir) and 0 < len(os.listdir(query_dir)):
                self.logger.info('Loading processed queries')
                for data_set in data_sets:
                    file_path = os.path.join(query_dir,
                                             'qa_{}.txt'.format(data_set))
                    saved_queries = self.load_saved_query(file_path)
                    data[data_set] = saved_queries
                return data

        # if no saved data file, generate one and save them
        self.logger.info(('Extracting knowledge base to generate'
                          ' entities and relations'))
        if self.triples is None:
            self.load_kb()
        if self.entities is None or self.relations is None:
            self.parse_kb()
        embedding_dir = os.path.dirname(self.config['entity_path'])
        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)
        with open(self.config['entity_path'], 'w') as f:
            for entity, idx in self.entities.items():
                f.write("{}\t{}\n".format(entity, idx))
        with open(self.config['relation_path'], 'w') as f:
            for relation, idx in self.relations.items():
                f.write("{}\t{}\n".format(relation, idx))

        # load embedding model
        self.logger.info('Loading embedding model: {}'.format(
                            self.config['embedding_method']))
        if 'Word2vec' == self.config['embedding_method']:
            model = Word2Vec.load(emb_path)
            model = model.wv
            # setup multi process
            multiprocessing.set_start_method('spawn')
            pool = multiprocessing.Pool(self.config['pool_size'])
        elif 'BERT' == self.config['embedding_method']:
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            model_path = os.path.join(config['embedding_bert'], 'fine_tune_model.bin')
            if os.path.exists(model_path):
                model = RobertaModel.from_pretrained(config['embedding_bert'])
            else:
                model = RobertaModel.from_pretrained('roberta-base')
            model.cuda()
            model.eval()

        for data_set in data_sets:
            self.logger.info('Processing {} set ...'.format(data_set))

            # load file and parse
            file_path = os.path.join(self.config['meta_dir'],
                                     '{}-hop'.format(n_hop),
                                     'vanilla',
                                     'qa_{}.txt'.format(data_set))
            queries = self.parse_query(file_path)

            # parallelly convert sentence to embedding
            if 'Word2vec' == self.config['embedding_method']:
                partial_word2vec = partial(parallel_word2vec, model=model)
                #processed_queries = pool.map(partial_word2vec, queries)
                processed_queries = list(tqdm(pool.imap(self.partial_word2vec, queries),
                                              total=len(queries)))
            elif 'BERT' == self.config['embedding_method']:
                processed_queries = self.sequential_bert(queries, tokenizer, model)

            # save all embeddings to a file
            dir_path = os.path.join(self.config['processed_query_dir'],
                                     '{}-hop'.format(n_hop))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path,
                                     'qa_{}.txt'.format(data_set))
            with open(file_path, 'w') as f:
                for processed_query in tqdm(processed_queries, desc='write to file'):
                    idx, subj_id, q_emb, ans_ids = processed_query
                    idx = str(idx)
                    subj_id = str(subj_id)
                    q_emb = ' '.join([str(x) for x in q_emb])
                    ans_ids = ' '.join([str(x) for x in ans_ids])
                    string = '\t'.join([idx, subj_id, q_emb, ans_ids])
                    f.write(string)
                    f.write('\n')

            data[data_set] = processed_queries

        if 'Wordwvec' == self.config['embedding_method']:
            pool.close()

        return data

    def load_kb_as_coo_matrixes(self):
        """parse knowledge graph and convert to COO matrixes
    
        Returns:
            M_subj(COO_Matrix): a sparse matrix with size (# of triples, # of entities)
            M_rel(COO_Matrix): a sparse matrix with size (# of triples, # of relation)
            M_obj(COO_Matrix): a sparse matrix with size (# of triples, # of entities)
        """
        # extract knowledge base, entities and relations
        read_entity = False
        if 'entity_path' in self.config and self.config['entity_path'] is not None:
            read_entity = True
            with open(self.config['entity_path'], 'r') as f:
                self.entities = {}
                for line in f.readlines():
                    entity, idx = line.split('\t')
                    idx = int(idx)
                    self.entities[entity] = idx
                entities_rev = [''] * len(self.entities)
                for k, v in self.entities.items():
                    entities_rev[v] = k
    
        read_relation = False
        if 'relation_path' in self.config and self.config['relation_path'] is not None:
            read_relation = True
            with open(self.config['relation_path'], 'r') as f:
                self.relations = {}
                for line in f.readlines():
                    relation, idx = line.split('\t')
                    idx = int(idx)
                    self.relations[relation] = idx

        # if no saved entity and relation files, parse kb and save them
        if not read_entity or not read_relation:
            self.load_kb()
            self.parse_kb()
            with open(self.config['entity_path'], 'w') as f:
                for entity, idx in self.entities.items():
                    f.write("{}\t{}\n".format(entity, idx))
            with open(self.config['relation_path'], 'w') as f:
                for relation, idx in self.relations.items():
                    f.write("{}\t{}\n".format(relation, idx))

            # for debugging
            if self.config['DEBUG']:
                entities_rev = [''] * len(self.entities)
                for k, v in self.entities.items():
                    entities_rev[v] = k
    
        # gather indexes for subject, relation, and object
        if self.triples is None:
            self.load_kb()
        total_triples = len(self.triples)
        subj_idx = []
        rel_idx = []
        obj_idx = []
        for idx, triple in enumerate(self.triples):
            s, r, o = triple
            subj_idx.append([idx, self.entities[s]])
            rel_idx.append([idx, self.relations[r]])
            obj_idx.append([idx, self.entities[o]])
            # add a reverse edge
            r_rev = '_'.join([r, 'rev'])
            subj_idx.append([idx + total_triples, self.entities[o]])
            rel_idx.append([idx + total_triples, self.relations[r_rev]])
            obj_idx.append([idx + total_triples, self.entities[s]])

        # convert to COO matrixes
        subj_data = torch.FloatTensor([1] * len(subj_idx))
        subj_idx = torch.LongTensor(subj_idx).T
        subj_size = [total_triples * 2, len(self.entities)] # add reverse edges
        rel_data = torch.FloatTensor([1] * len(rel_idx))
        rel_idx = torch.LongTensor(rel_idx).T
        rel_size = [total_triples * 2, len(self.relations)] # add reverse edges
        obj_data = torch.FloatTensor([1] * len(obj_idx))
        obj_idx = torch.LongTensor(obj_idx).T
        obj_size = [total_triples * 2, len(self.entities)] # add reverse edges
    
        M_subj = torch.sparse.FloatTensor(subj_idx, subj_data, torch.Size(subj_size))
        M_rel = torch.sparse.FloatTensor(rel_idx, rel_data, torch.Size(rel_size))
        M_obj = torch.sparse.FloatTensor(obj_idx, obj_data, torch.Size(obj_size))
    
        # for debugging
        if self.config['DEBUG']:
            return M_subj, M_rel, M_obj, entities_rev
        return M_subj, M_rel, M_obj

if __name__ == '__main__':
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    # substitute_embedding(config)
    # metaqa = MetaQAReader(config)
    # data = metaqa.load_query()
    # M_subj, M_rel, M_obj, X_rev = metaqa.load_kb_as_coo_matrixes()

    reader = MetaQAReader(config)
    #reader.load_kb_as_coo_matrixes()
    data = reader.load_query()
    print(data.keys())

