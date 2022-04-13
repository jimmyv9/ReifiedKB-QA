import os
import sys
import gensim.models
from gensim.test.utils import datapath
from gensim import utils
import re

class MyCorpus:
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        os.chdir(".")
        corpus_path = datapath(self.path)
        for line in open(corpus_path):
            line = line.split("\t")
            question, answer = line[0], line[1]
            question = tokenize_question(question)
            answer = tokenize_answer(answer)
            line = " ".join([question, answer])
            yield utils.simple_preprocess(line)


def tokenize_question(question):
    match = re.search("\[(.*)\]", question)
    entity = match.group()
    new_entity = "_".join(entity[1:-1].split())
    question = question.replace(entity, new_entity)
    return question

def tokenize_answer(answer):
    answer = answer.split("|")
    for i, ans in enumerate(answer):
        answer[i] = "_".join(ans.split())
    return " ".join(answer)

def main():
    if len(sys.argv) != 3:
        print("Execute: python create_word2vec_model.py file_to_embed model_outpath")
    to_embed = sys.argv[1]
    outfile = sys.argv[2]
    filename = os.path.realpath(to_embed)
    sentences = MyCorpus(filename)
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=128)
    model.save(outfile)
    return


if __name__ == "__main__":
    main()
