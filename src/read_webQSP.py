import json
import yaml
import gzip
import re
import os
import math
from tqdm import tqdm

def extract_json_entities(infile):
    """
    Extracts entities from WebQSP, mapping their MID in Freebase to their full
    name
    Input:
        infile (str): path to WebQSPs json training file
    Output:
        entities (dict): mapping of entity MIDs to their names
    """
    with open(infile, 'r') as fin:
        train_json = json.load(fin)
    questions = train_json['Questions']
    entities = dict()
    qents = set()
    aents = set()
    value = 0
    for q in questions:
        for p in q['Parses']:
            if p['TopicEntityMid'] == None:
                continue
            if p['AnnotatorComment']['Confidence'] != "Normal":
                continue
            subject = p['TopicEntityName']
            mid = p['TopicEntityMid']
            entities[mid] = subject
            qents.add(mid)
            for ans in p['Answers']:
                if ans['AnswerType'] == "Entity":
                    mid = ans['AnswerArgument']
                    entities[mid] = ans['EntityName']
                    aents.add(mid)
                else:
                    # Answer is a value
                    val_id = 'val_' + str(value)
                    entities[val_id] = ans['AnswerArgument']
                    aents.add(val_id)
                    value += 1
    return entities, qents, aents

def extract_json_relations(infile):
    """
    Extracts relations from webQSP json files
    Input:
        infile (str): path to webQSP json file
    Ouput:
        rels (set): relations found in file
    """
    with open(infile, 'r') as fin:
        train_json = json.load(fin)
    questions = train_json['Questions']
    rels = set()
    for q in questions:
        for p in q['Parses']:
            if p['InferentialChain'] != None:
                for rel in p['InferentialChain']:
                    if rel != None:
                        rels.add(rel)
    return rels

def extract_ntriple_entities(infile):
    """
    Extract entity MIDs from N-triple RDF formatted file
    """
    entities = set()
    s_ents = set()
    o_ents = set()
    with open(infile, 'r') as fin:
        for triple in fin:
            subj, rel, obj, _ = triple.strip()[:-1].split('\t')
            s_ent = subj.replace('>', '').split('/')[-1]
            entities.add(s_ent)
            s_ents.add(s_ent)
            if obj[0] == '<':
                o_ent = obj.replace('>', '').split('/')[-1]
                entities.add(o_ent)
                o_ents.add(o_ent)
    return entities, s_ents, o_ents

def generate_questions(infile, entities, outfile):
    """
    Generate training/testing file similar to the one used by MetaQA where each
    line looks like:
        "this here is a question with [subj_entity]? \t answer1|answer2|answer3"
    Input:
        infile (str): path to a WebQSP json file
        outfile (file obj): a file where all questions are written
        test (bool): optional - indicates whether this is a testing file, if so
        there is no need to split into train/dev
        split (float): optional - indicates the train/dev split for our dataset
    """
    with open(infile, 'r') as fin:
        json_oi = json.load(fin)
        all_questions = []
        questions = json_oi['Questions']
        nlq_answer = dict()
        for q in questions:
            # For each question, find all the possible parses to the question, and
            # append answers to it accordingly
            processed = q['ProcessedQuestion']
            for p in q['Parses']:
                if p['AnnotatorComment']['Confidence'] != "Normal" or p['TopicEntityName'] == None:
                    # Avoid parses with "Low" or "VeryLow" confidence, avoid
                    # parses that have no topic entity (only 1 in train set)
                    continue
                potential = p['PotentialTopicEntityMention']
                mid = p['TopicEntityMid']
                if mid not in entities:
                    continue
                entity = '[' + entities[mid] + ']'
                nlq_parse = re.sub(potential, entity, processed)
                answers = set()
                for ans in p['Answers']:
                    if ans['AnswerType'] == "Value":
                        answers.add(ans['AnswerArgument'])
                    elif ans['AnswerType'] == "Entity":
                        if ans["AnswerArgument"] not in entities:
                            continue
                        answer = entities[ans["AnswerArgument"]]
                        answers.add(ans['EntityName'])
                    if nlq_parse in nlq_answer:
                        new_set = nlq_answer[nlq_parse].union(answers)
                        nlq_answer[nlq_parse] = new_set
                    else:
                        if len(answers) == 0:
                            continue
                        nlq_answer[nlq_parse] = answers
    with open(outfile, 'w') as fout:
        for nlq in nlq_answer:
            answers = list(nlq_answer[nlq])
            line = nlq + '\t' + '|'.join(answers) + '\n'
            fout.write(line)

def split_qa(train, dev):
    with open(train, 'r') as fin:
        all_lines = fin.readlines()
    size = len(all_lines)
    cut = math.floor(0.9*size)
    train_set = all_lines[:cut]
    dev_set = all_lines[cut:]
    with open(train, 'w') as fout:
        for qa in train_set:
            fout.write(qa)
    with open(dev, 'w') as fout:
        for qa in dev_set:
            fout.write(qa)

def subset_freebase(fb_path, entities, relations, out_path):
    """
    Subsets the 1-hop neighborhood of entities within freebase
    """
    n = 0
    t = tqdm(total=3100000000)
    with gzip.open(fb_path, 'r') as fb:
        with open(out_path, 'w') as fout:
            to_write = ""
            for idx, triple in enumerate(fb):
                triple = triple.decode()
                subj, rel, obj, _ = triple.strip()[:-1].split('\t')
                relation = rel.replace('>', '').split('/')[-1]
                mid = subj.replace('>', '').split('/')[-1]
                if mid in entities and relation in relations and is_en(obj):
                    to_write += triple
                    #fout.write(triple)
                if idx % 10000000 == 0:
                    fout.write(to_write)
                    to_write = ""
                    t.update(10000000)
            fout.write(to_write)
            t.update(idx % 10000000)

def is_en(obj):
    """
    Verifies if an object from a triple is an object
    Input:
        obj (str): object
    Output:
        is_en (bool): true if object is in english or a number, date, etc.
        false otherwise
    """
    if obj[0] == '"':
        # Object is a string and could be in any language
        parsed_obj = obj.split('@')
        if len(parsed_obj) == 1:
            return True
        language = parsed_obj[-1]
        if language not in ['en', 'en-US', 'en-GB']:
            return False
    return True

def filter_kg_by_relations(kg, rels, outfile):
    """
    Remove all triples that do not have a relation found in our training data
    Input:
        kg (str): path to a text file with our triples
        rels (str): set of relations found in WebQSP training set
        outfile (str): path to output file
    """
    t = tqdm(total=11489425)
    with open(kg, 'r') as fin:
        with open(outfile, 'w') as fout:
            for idx, triple in enumerate(fin):
                subj, rel, obj, _ = triple.strip()[:-1].split('\t')
                rel = rel.replace('>', '').split('/')[-1]
                if rel in rels and is_en(obj):
                    fout.write(triple)
                if idx % 100000 == 0:
                    t.update(100000)
    t.update(89425)

def filter_kg_by_entities(kg, to_remove, outfile):
    """
    Remove all triples which subject or object are included in our remove list
    Input:
        kg (str): path to a text file with our triples
        to_remove (str): set of entities MIDs
        outfile (str): path to output file
    """
    with open(kg, 'r') as fin:
        with open(outfile, 'w') as fout:
            for idx, triple in enumerate(fin):
                subj, rel, obj, _ = triple.strip()[:-1].split('\t')
                subject = subj.replace('>', '').split('/')[-1]
                if subject not in to_remove and obj[0] == '"':
                    fout.write(triple)
                    continue
                objct = obj.replace('>', '').split('/')[-1]
                if subject not in to_remove and objct not in to_remove:
                    fout.write(triple)

def extract_cvts(cvt_file):
    # Extract CVTs
    cvts = set()
    with open(cvt_file, 'r') as fin:
        for cvt in fin:
            cvt = cvt.strip()
            cvts.add(cvt)
    return cvts

def get_named_entities(infile):
    """
    Given an n-triple file, return a mapping of named entity MIDs to their
    respective named entity name
    """
    named_entities = {}
    with open(infile, 'r') as fin:
        for triple in fin:
            subj, rel, obj, _ = triple.strip().split('\t')
            relation = rel.replace('>', '').split('/')[-1]
            if relation == 'type.object.name':
                subject = subj.replace('>', '').split('/')[-1]
                ne = obj.replace('"', '').split('@')[0]
                named_entities[subject] = ne
    return named_entities

def exhaust_cvts(infile, cvts):
    """
    Finds which entities must not belong to the set of CVTs, this is possible
    since two CVTs may never connect to one another
    Input:
        infile (str): path to N-triple RDF formatted file
        cvts (set): contains the MIDs for CVT candidates
    Output:
        not_cvts (set): pairs of nodes which are unlikely to be CVTs
    """
    not_cvts = set()
    with open(infile, 'r') as fin:
        for triple in fin:
            subj, rel, obj, _ = triple.strip().split('\t')
            subject = subj.replace('>', '').split('/')[-1]
            if obj[0] == '"':
                continue
            objct = obj.replace('>', '').split('/')[-1]
            if subject in cvts and objct in cvts:
                not_cvts.add(subject)
                not_cvts.add(objct)
    return not_cvts

def get_edge_relations(infile, cvts):
    """
    Generates a list of edge relations, where:
        0 = CVT -> CVT
        1 = entity -> entity
        2 = entity -> CVT
        3 = CVT -> entity
    """
    relation_types = []
    with open(infile, 'r') as fin:
        for triple in fin:
            subj, rel, obj, _ = triple.strip().split('\t')
            subj_is_cvt = subj.replace('>', '').split('/')[-1] in cvts
            if obj[0] == '"':
                obj_is_cvt = False
            else:
                obj_is_cvt = obj.replace('>', '').split('/')[-1] in cvts

            if subj_is_cvt and obj_is_cvt:
                # r_CVT -> r_CVT
                relation_types.append(0)
            if not subj_is_cvt and not obj_is_cvt:
                # r_E -> r_CVT
                relation_types.append(1)
            elif not subj_is_cvt and obj_is_cvt:
                # r_CVT -> r_E
                relation_types.append(2)
            elif subj_is_cvt and not obj_is_cvt:
                # r_E -> r_E
                relation_types.append(3)
    return relation_types

def beautify_kb(infile, entities, cases, outfile):
    """
    Makes KB look like MetaQA's KB
    Input:
        infile (str): n-triple WebQSP input file
        entities (dict): mapping of MIDs to entity names
        outfile (str): path to output file
    """
    # Make entities
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            for idx, triple in enumerate(fin):
                case = cases[idx]
                subj, rel, obj, _ = triple.strip().split('\t')
                smid = subj.replace('>', '').split('/')[-1]
                if smid in entities:
                    smid = entities[smid]
                if obj[0] == '"':
                    omid = obj.replace('"', '').split('@')[0]
                    omid = omid.split('^^')[0] # to deal with dates
                else:
                    omid = obj.replace('>', '').split('/')[-1]
                    if omid in entities:
                        omid = entities[omid]
                relation = rel.replace('>', '').split('/')[-1] + '_' + str(case)
                smid = smid.replace('|', '')
                omid = omid.replace('|', '')
                fout.write(smid+'|'+relation+'|'+omid+'\n')


def main(config):
    entities, qmids, amids = extract_json_entities(config['train_path_wqsp'])
    relations = extract_json_relations(config['train_path_wqsp'])
    # Extract CVT ids
    cvt_file = "all_webqsp_cvts.txt" # stores all CVT MIDs
    cvts = extract_cvts(cvt_file)

    # Stores hops from all entities 
    raw_subset = "freebase_subset_all.txt"
    # Only stores hops from entities appearing in train questions
    qents_subset = "freebase_subset_qids.txt"


    raw_2hop = "freebase_2hop_all.txt"

    filter_subset = "freebase_subset_filter.txt"
    qfilter_subset = "freebase_subset_qfilter.txt"
    filter_2hop = "freebase_subset_2hop_filter.txt"
    final = config['kb_path_wqsp']

    # The function `subset_freebase` runs on 31B triples, and will take 4-6
    # hours to execute total, do not uncomment them unless you are getting data
    # for the first time and want to recompute

    # subset_freebase(config['freebase'], entities, relations, raw_subset)
    # subset_freebase(config['freebase'], qmids, relations, qents_subset)

    # Extract entities from filter
    # ents_1hop, smids_1hop, omids_1hop = extract_ntriple_entities(filter_subset)
    # qents_1hop, qsmids_1hop, qomids_1hop = extract_ntriple_entities(qfilter_subset)

    # new_1hop_ents = ents_1hop.difference(set(entities.keys()))
    # print("We found {} nodes after 1-hop".format(len(new_1hop_ents)))

    # Get 2-hop neighborhood
    # subset_freebase(config['freebase'], ents_1hop, relations, raw_2hop)
    ents_2hop, subj_mids, obj_mids = extract_ntriple_entities(raw_2hop)
    ne_2hop_dict = get_named_entities(raw_2hop)
    # Make all named entities have unique values
    ne_vals = set()
    unique_id = 0
    for key in ne_2hop_dict:
        val = ne_2hop_dict[key]
        if val in ne_vals:
            ne_2hop_dict[key] = val + "_" + str(unique_id)
            unique_id += 1
        ne_vals.add(ne_2hop_dict[key])
    ne_2hop = set(ne_2hop_dict.keys()) # get all entity MIDs in a set
    ne_og = set(entities.keys()).intersection(ne_2hop)
    unnamed_og = set(entities.keys()).difference(ne_og)
    # Get set of CVTs based on whether the freebase entity is named
    cvt_set = subj_mids.difference(ne_2hop).difference(unnamed_og)
    # Determine which of our current CVTs have linking edges, and remove them
    # from the KG
    # not_cvts = exhaust_cvts(raw_2hop, cvt_set)
    # filter_kg_by_entities(raw_2hop, not_cvts, filter_2hop)

    generate_questions(config['train_path_wqsp'], ne_2hop_dict, config['train_path_proc'])
    split_qa(config['train_path_proc'], config['dev_path_proc'])
    generate_questions(config['test_path_wqsp'], ne_2hop_dict, config['test_path_proc'])

    # Find relationship types between all leftover edges and "beautify" the KB
    edge_relations = get_edge_relations(filter_2hop, cvt_set)
    beautify_kb(filter_2hop, ne_2hop_dict, edge_relations, final)
    return

if __name__ == '__main__':
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    main(config)
