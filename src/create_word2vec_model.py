import gensim.models
from gensim.test.utils import datapath
from gensim import utils
import re

class MyCorpus:
    def __iter__(self):
        corpus_path = datapath("/Users/JPV/Documents/UT_Dallas/2022 Spring/NLP Research/ReifiedKB-QA/data/MetaQA/1-hop/vanilla/qa_train.txt")
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
    sentences = MyCorpus()
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=128)
    model.save("word2vec-metaQA-1hop")
    return


if __name__ == "__main__":
    main()
