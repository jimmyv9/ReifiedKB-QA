import os
import yaml
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    """The whole training and validate process
    """
    # prepare saving environment
    save_dir = os.path.join(config['model_save_path'], config['model_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_dir = config['logger_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_dir = config['tensorboard_path']
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    log_file_name = 'log_{}.txt'.format(config['model_name'])
    logger = get_logger(os.path.join(config['logger_dir'], log_file_name)) # for logger
    writer = SummaryWriter(config['tensorboard_path']) # for tensorboard

    # check cuda
    gpus = []
    if config['use_cuda']:
        if torch.cuda.is_available():
            # get gpu ids
            gpus = [int(gpu) for gpu in config['gpus'].split(',')]
            if 1 == len(gpus):
                device = torch.device('cuda:{}'.format(gpus[0]))
            logger.info('Running on CUDA {}'.format(config['gpus']))
        else:
            device = torch.device('cpu')
            logger.warning('Cannot find CUDA, using CPU')
    else:
        device = torch.device('cpu')
        logger.info('Running on CPU')


    # read files
    logger.info("Read files and prepare data")

    # read KB, get M_subj, M_rel, M_obj
    M_subj, M_rel, M_obj = read_KB(config['kb_path'])
    if 1 >= len(gpus):
        M_subj = M_subj.to(device)
        M_rel = M_rel.to(device)
        M_obj = M_obj.to(device)
    else:
        M_subj = M_subj.cuda()
        M_rel = M_rel.cuda()
        M_obj = M_obj.cuda()
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # for train set
    data = read_metaqa(config['emb_path'])
    metaqa_train = MetaQADataset(data, M_subj.size(-1))
    train_dataloader = DataLoader(dataset=metaqa_train, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # for dev/validation set
    data = read_metaqa(config['emb_path'])
    metaqa_dev = MetaQADataset(data, M_subj.size(-1))
    dev_dataloader = DataLoader(dataset=metaqa_dev, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, optimizer, criterion = make_model(config, [M_subj, M_rel, M_obj])
    if 1 >= len(gpus):
        model.to(device)
    else:
        model = nn.DataParallel(model, device_ids=gpus)
        model = model.cuda()

    train_losses = []
    dev_losses = []
    best_train_loss = 1e9
    best_dev_loss = 1e9

    logger.info("Start training")
    
    model.train()
    for ep in range(config['MAX_TRAIN_EPOCH']):
        for batch_idx, data in enumerate(train_dataloader):
            # forward
            inputs, y = data # inputs: x, q
            if 1 >= len(gpus):
                inputs = [x.to(device) for x in inputs]
                y = y.to(device)
            else:
                inputs = [x.cuda() for x in inputs]
                y = y.cuda()
            if 'kb_multihop' == config['task']:
                y_hat = model(*inputs)
            else:
                raise ValueError

            # calculation loss
            loss = criterion(y_hat, y)
            loss_value = loss.item()
            train_losses.append(loss_value)
            writer.add_scalar('train loss', loss_value,
                                ep * len(train_dataloader) + batch_idx)

            # for statistics
            if loss_value < best_train_loss:
                best_train_loss = loss_value

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Train Epoch:[{}/{}]\t loss={:.3f}".format(
                        ep + 1, config['MAX_TRAIN_EPOCH'], np.mean(train_losses)))

        # validation/development set
        if (ep + 1) % config['DEV_EPOCH'] == 0:
            model.eval()
            dev_loss = []
            for batch_idx, data in enumerate(dev_dataloader):
                inputs, y = data # inputs: x, q
                inputs = [x.to(device) for x in inputs]
                y = y.to(device)
                if 'kb_multihop' == config['task']:
                    y_hat = model(*inputs)
                else:
                    raise ValueError
                loss = criterion(y_hat, y)
                loss_value = loss.item()
                dev_loss.append(loss_value)
                writer.add_scalar('dev loss raw', loss_value,
                                    (((ep + 1) / config['DEV_EPOCH'] - 1) *
                                    len(dev_dataloader) + batch_idx))

            dev_loss_avg = np.mean(dev_loss)
            writer.add_scalar('dev loss', dev_loss_avg, (ep + 1) / config['DEV_EPOCH'])
            dev_losses.append(dev_loss_avg)
            # for statistics
            if dev_loss_avg < best_dev_loss:
                best_dev_loss = dev_loss_avg

                # save the best model based on the dev/validation set
                if 1 >= len(gpus):
                    model.cpu()
                    state = {'epoch': ep,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, os.path.join(config['model_save_path'],
                                                   config['model_name'],
                                                   'best_model'))
                    model.to(device)
                else:
                    state = {'epoch': ep,
                             'state_dict': model.module.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state, os.path.join(config['model_save_path'],
                                                   config['model_name'],
                                                   'best_model'))

            logger.info("Dev Epoch:[{}]\t loss={:.3f}".format(
                            (ep + 1) // config['DEV_EPOCH'], np.mean(dev_loss)))

            model.train()

    # save the final model
    if 1 >= len(gpus):
        model.cpu()
        state = {'epoch': ep,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
    else:
        state = {'epoch': ep,
                 'state_dict': model.module.state_dict(),
                 'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(config['model_save_path'],
                                   config['model_name'],
                                   'final_model'))

    # save the losses in text to prevent error from tensorboard
    #save_dir = os.path.join(config['model_save_path'], config['model_name'])
    with open(os.path.join(save_dir, 'train_loss.txt'), 'w') as f:
        train_losses_str = ' '.join([str(loss) for loss in train_losses])
        f.write(train_losses_str)
    with open(os.path.join(save_dir, 'dev_loss.txt'), 'w') as f:
        dev_losses_str = ' '.join([str(loss) for loss in dev_losses])
        f.write(dev_losses_str)

    logger.info("Finish training")

    # how to output results. set a number? or select the top k?

if __name__ == "__main__":
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    run(config)

