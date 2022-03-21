import logging
import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter
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
    logger.addHandler(fh)

    if file_name is not None and '' != file_name:
        fh = logging.FileHandler(file_name, mode='w')  # file handler, mode rewrite
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(ch)

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
    if 'kb_multihop' == config['hidden_size']:
        model = RefiedKBQA(config['N_W2V'], config['N_R'], kb_info)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def run(config):
    """The whole training, validate, test process
    """
    logger = get_logger(config['logger_file_name']) # for logger
    writer = SummaryWriter(config['tensorboard_path']) # for tensorboard

    # read files
    logger.info("Read files")
    # add config['N_R'] here

    # process labels
    logger.info("Process questions and the knowledge base")

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, optimizer, criterion = make_model(config, [M_subj, M_rel, M_obj])

    train_loss = []
    best_loss = 1e9

    logger.info("Start training")
    
    model.train()
    for ep in range(config['MAX_TRAIN_EPOCH']):
        for batch_idx, data in enumerate(train_dataloader):
            # forward
            x, q, n_hop, y = data
            if 'kb_multihop' == config['task']:
                x = model(x, q, n_hop)
            else:
                raise ValueError

            # calculation loss
            loss = criterion(x, y)
            loss_value = loss.item()
            train_loss.append(loss_value)
            writer.add_scalar('train loss', loss_value,
                                ep * len(train_dataloader) + batch_idx)
            if loss_value < best_loss:
                best_loss = loss_value

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Train Epoch:[{}/{}]\t loss={:.3f}".format(
                        ep + 1, config['MAX_TRAIN_EPOCH'], torch.mean(train_loss)))

        # validation/development set
        if (ep + 1) % config['DEV_EPOCH'] == 0:
            model.eval()
            dev_loss = []
            for batch_idx, data in enumerate(dev_dataloader):
                x, q, n_hop, y = data
                for i in range(n_hop):
                    x = model(x, q, i, M_subj, M_rel, M_obj)
                loss = criterion(x, y)
                loss_value = loss.item()
                dev_loss.append(loss_value)
                writer.add_scalar('dev loss', loss_value,
                                    (((ep + 1) / config['DEV_EPOCH'] - 1) *
                                    len(dev_dataloader) + batch_idx))
            logger.info("Dev Epoch:[{}]\t loss={:.3f}".format(
                            (ep + 1) / config['DEV_EPOCH'], torch.mean(dev_loss)))

    logger.info("Finish training")

    # how to output results. set a number? or select the top k?

if __name__ == "__main__":
    with open("../config.yml") as f:
        config = yaml.safe_load(f)
    run(config)
