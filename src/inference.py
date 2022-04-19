import os
import yaml
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from read_metaQA import read_metaqa, read_KB, MetaQADataset
from model import *
from main import get_logger, make_model, collate_fn

def run(config):
    """The whole test process
    """
    # prepare saving environment
    save_dir = os.path.join(config['model_save_path'], config['model_name'])
    if not os.path.exists(save_dir):
        raise ValueError('Wrong model path')

    log_dir = config['logger_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_name = 'log_{}_infer.txt'.format(config['model_name'])
    logger = get_logger(os.path.join(config['logger_dir'], log_file_name)) # for logger

    # check cuda
    if config['use_cuda']:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(config['gpu']))
            logger.info('Running on CUDA')
        else:
            device = torch.device('cpu')
            logger.warning('Cannot find CUDA, using CPU')
    else:
        device = torch.device('cpu')
        logger.info('Running on CPU')

    # read files
    logger.info("Read files and prepare data")

    # read KB, get M_subj, M_rel, M_obj
    M_subj, M_rel, M_obj, X_rev = read_KB(config)
    M_subj.requires_grad = False
    M_rel.requires_grad = False
    M_obj.requires_grad = False
    M_subj = M_subj.to(device)
    M_rel = M_rel.to(device)
    M_obj = M_obj.to(device)
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # for test set
    data = read_metaqa(config['emb_path'])
    metaqa_test = MetaQADataset(data, M_subj.size(-1))
    test_dataloader = DataLoader(dataset=metaqa_test,
                                 batch_size=config['batch_size'],
                                 shuffle=True,
                                 collate_fn=collate_fn)

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, _, _ = make_model(config, [M_subj, M_rel, M_obj])
    # load pre-trained data
    state = torch.load(os.path.join(save_dir, config['read_from']))
    model.load_state_dict(state['state_dict'])
    model.to(device)

    logger.info('Dense 1 weight')
    logger.info(str(model.dense1.weight))
    logger.info('Dense 1 bias')
    logger.info(str(model.dense1.bias))

    scores = []
    results = []
    mrr_scores = []
    mrr_results = []

    logger.info("Start testing")

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            q_idx, inputs, y = data # inputs: [x, q]
            inputs = [x.to(device) for x in inputs]
            if 'kb_multihop' == config['task']:
                y_hat, _ = model(*inputs)
            else:
                raise ValueError
            y_sort, y_idx = torch.sort(y_hat, dim=-1, descending=True)
            y_idx = y_idx[:, :5].tolist()
            for y_pred, y_true in zip(y_idx, y):
                # accuracy @1
                if 0 < y_true[y_pred[0]].item():
                    scores.append(1)
                    results.append(y_pred[0])
                else:
                    scores.append(0)
                    results.append(-1)

                # MRR
                score = 0
                for i, value in enumerate(y_pred):
                    if 0 < y_true[value].item():
                        score = 1/(i + 1)
                        mrr_results.append(value)
                        break
                mrr_scores.append(score)
                if 0 == score:
                    mrr_results.append(-1)

        logger.info('Test accuracy @1: {:.3f} MRR@5: {:.3f}'.format(np.mean(scores), np.mean(mrr_scores)))

    entity = []
    for result in results:
        if -1 == results:
            entity.append('')
        else:
            entity.append(X_rev[result])
    mrr_entity = []
    for result in mrr_results:
        if -1 == results:
            mrr_entity.append('')
        else:
            mrr_entity.append(X_rev[result])

if __name__ == '__main__':
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    run(config)

