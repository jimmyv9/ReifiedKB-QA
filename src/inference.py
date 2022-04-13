import os
import yaml
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from read_metaQA import read_metaqa, read_KB, MetaQADataset
from model import *

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

def make_model(config, kb_info):
    """Build model, initalize optimizer and criterion
    Args:
        config(dict): from yaml
        kb_info: M_subj, M_rel, M_obj matrixes where
            M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                             triples in the KB
            M_rel(matrix): dim (N_T, N_R)
            M_obj(matrix): dim (N_T, N_E)
    """
    if 'kb_multihop' == config['task']:
        model = RefiedKBQA(config['N_W2V'], config['N_R'], config['n_hop'], kb_info)
    config['lr'] = float(config['lr'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    #criterion = nn.CrossEntropyLoss()
    criterion = WeightedSoftmaxCrossEntropyLoss()

    return model, optimizer, criterion

def collate_fn(batch_data):
    """Gather a batch of data and convert them to proper format
    Args:
        batch_data(list): a list of data with length equal to the batch size

    Return:
        [xs, qs](Tensor, Tensor): two tensors both with size (batch_size, 300)
        labels(Tensor): a tensor with size (batch_size,)
    """
    # [[[batch_data], total numbers of entities], [], ...] -> [batch_data], [total numbers of entities]
    batch_data = list(zip(*batch_data))
    batch_data, total_number = batch_data
    total_number = total_number[0]

    # [[[x, q], label], [], ...] -> [[x, q], [], ...], [label]
    batch_data = list(zip(*batch_data))
    inputs, labels = batch_data

    # [[x, q], [], ...] -> [x], [q]
    xs, qs = list(zip(*inputs))

    # convert list to tensor
    xs = torch.LongTensor(xs)
    xs = F.one_hot(xs, num_classes=total_number).type(torch.FloatTensor)
    qs = torch.cat(qs, dim=0)

    # convert list of label to uniform distribution on labels
    label_tensor = torch.zeros_like(xs, dtype=torch.float)
    for idx, label in enumerate(labels):
        label = torch.tensor(label)
        label_tensor[idx,:].index_fill_(0, label, 1 / (len(label)))
    labels = label_tensor

    # freeze for training
    xs.requires_grad = False
    qs.requires_grad = False
    labels.requires_grad = False

    return [xs, qs], labels

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
            gpus = [int(gpu) for gpu in config['gpus'].split(',')]
            device = torch.device('cuda:{}'.format(gpus[0]))
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
    M_subj = M_subj.to(device)
    M_rel = M_rel.to(device)
    M_obj = M_obj.to(device)
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # for test set
    data = read_metaqa(config['emb_path'])
    metaqa_test = MetaQADataset(data, M_subj.size(-1))
    test_dataloader = DataLoader(dataset=metaqa_test,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=collate_fn)

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, _, _ = make_model(config, [M_subj, M_rel, M_obj])
    # load pre-trained data
    state = torch.load(os.path.join(save_dir, config['read_from']))
    model.load_state_dict(state['state_dict'])
    model.to(device)

    scores = []
    results = []
    mrr_scores = []
    mrr_results = []

    logger.info("Start testing")

    model.eval()
    for data in tqdm(test_dataloader):
        inputs, y = data
        inputs = [x.to(device) for x in inputs]
        if 'kb_multihop' == config['task']:
            y_hat, _ = model(*inputs)
        else:
            raise ValueError
        #y_hat = torch.argmax(y_hat, dim=1)
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

