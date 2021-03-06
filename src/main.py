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

    # [[q_id, x, q_emb, label], [], ...] -> [q_id], [x], [q_emb], [label]
    batch_data = list(zip(*batch_data))
    q_idx, xs, qs, labels = batch_data


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

    return q_idx, [xs, qs], labels

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

    tensorboard_dir = os.path.join(config['tensorboard_path'], config['model_name'])
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    log_file_name = 'log_{}.txt'.format(config['model_name'])
    logger = get_logger(os.path.join(config['logger_dir'], log_file_name)) # for logger
    writer = SummaryWriter(tensorboard_dir) # for tensorboard

    # check cuda
    if config['use_cuda']:
        if torch.cuda.is_available():
            # get gpu id
            device = torch.device('cuda:{}'.format(config['gpu']))
            logger.info('Running on CUDA {}'.format(config['gpu']))
        else:
            device = torch.device('cpu')
            logger.warning('Cannot find CUDA, using CPU')
    else:
        device = torch.device('cpu')
        logger.info('Running on CPU')


    # read files
    logger.info("Read files and prepare data")

    # read KB, get M_subj, M_rel, M_obj
    #M_subj, M_rel, M_obj, X_rev = read_KB(config) # for debugging
    M_subj, M_rel, M_obj, X_rev = read_KB(config) # for debugging
    M_subj.requires_grad = False
    M_rel.requires_grad = False
    M_obj.requires_grad = False
    M_subj = M_subj.to(device)
    M_rel = M_rel.to(device)
    M_obj = M_obj.to(device)
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # for train set
    data = read_metaqa(config['train_emb_path'])
    metaqa_train = MetaQADataset(data, M_subj.size(-1))
    train_dataloader = DataLoader(dataset=metaqa_train,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # for dev/validation set
    data = read_metaqa(config['dev_emb_path'])
    metaqa_dev = MetaQADataset(data, M_subj.size(-1))
    dev_dataloader = DataLoader(dataset=metaqa_dev,
                                batch_size=config['batch_size'],
                                shuffle=True,
                                collate_fn=collate_fn)

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, optimizer, criterion = make_model(config, [M_subj, M_rel, M_obj])
    model.to(device)

    train_losses = []
    dev_losses = []
    best_train_loss = 1e9
    best_dev_loss = 1e9

    logger.info("Start training")
    
    model.train()
    for ep in range(config['MAX_TRAIN_EPOCH']):
        train_loss = []
        train_results = []
        for batch_idx, data in enumerate(train_dataloader):
            # forward
            _, inputs, y = data # inputs: x, q
            inputs = [x.to(device) for x in inputs]
            y = y.to(device)
            if 'kb_multihop' == config['task']:
                #y_hat = model(*inputs)
                y_hat, r = model(*inputs) # for debugging
            else:
                raise ValueError

            # calculation loss
            loss = criterion(y_hat, y)
            # for record
            loss_value = loss.item()
            train_losses.append(loss_value)
            train_loss.append(loss_value)
            writer.add_scalar('train loss', loss_value,
                                ep * len(train_dataloader) + batch_idx)

            # accuracy @1
            with torch.no_grad():
                y_hat = torch.argmax(y_hat, dim=-1).tolist()
                for y_pred, y_true in zip(y_hat, y):
                    if 0 < y_true[y_pred].item():
                        train_results.append(1)
                    else:
                        train_results.append(0)

            # for statistics
            if loss_value < best_train_loss:
                best_train_loss = loss_value

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Train Epoch:[{}/{}]\tloss={:.3f}\tacurracy={:.3f}".format(
                        ep + 1,
                        config['MAX_TRAIN_EPOCH'],
                        np.mean(train_loss),
                        np.mean(train_results)))

        # validation/development set
        if (ep + 1) % config['DEV_EPOCH'] == 0:
            model.eval()
            with torch.no_grad():
                dev_loss = []
                dev_results = []
                for batch_idx, data in enumerate(dev_dataloader):
                    _, inputs, y = data # inputs: x, q
                    inputs = [x.to(device) for x in inputs]
                    y = y.to(device)
                    if 'kb_multihop' == config['task']:
                        #y_hat = model(*inputs)
                        y_hat, r = model(*inputs) # for debugging
                    else:
                        raise ValueError

                    # for debugging
                    #with torch.no_grad():
                    #    x_idx = torch.argmax(inputs[0][0,:]).cpu().detach().item()
                    #    print('x:', x_idx, X_rev[x_idx], flush=True)
                    #    idxes = []
                    #    M = model.M_subj.cpu().to_dense()
                    #    for idx, triple in enumerate(M):
                    #        if 1 == triple[x_idx]:
                    #            print('x index in kb', idx, flush=True)
                    #            idxes.append(idx)
                    #    x_t = torch.sparse.mm(model.M_subj, inputs[0][:1,:].T) # (N_T, batch_size) dense
                    #    for idx in idxes:
                    #        print(x_t[idx], flush=True)
                    #    r = r[:1,:]
                    #    print('r:', r, flush=True)
                    #    r_t = torch.sparse.mm(model.M_rel, r.T) # (N_T, batch_size) dense
                    #    print('r_cal', r_t.T, flush=True)
                    #    tmp = (x_t * r_t).T
                    #    print('x_t * r_t', tmp)
                    #    print(torch.amax(tmp), torch.argmax(tmp))
                    #    y_hat = y_hat[:1, :]
                    #    print('y_hat:', y_hat, flush=True)
                    #    y_idx = torch.argmax(y_hat)
                    #    print('y_idx', y_idx, X_rev[y_idx])
                    #    for name, param in model.named_parameters():
                    #        if 'dense1' == name[:6]:
                    #            print(name, param, param.requires_grad)
                    #    input()

                    loss = criterion(y_hat, y)
                    loss_value = loss.item()
                    dev_loss.append(loss_value)
                    writer.add_scalar('dev loss raw', loss_value,
                                        (((ep + 1) / config['DEV_EPOCH'] - 1) *
                                        len(dev_dataloader) + batch_idx))

                    # accuracy @1
                    y_hat = torch.argmax(y_hat, dim=-1).tolist()
                    for y_pred, y_true in zip(y_hat, y):
                        if 0 < y_true[y_pred].item():
                            dev_results.append(1)
                        else:
                            dev_results.append(0)

            dev_loss_avg = np.mean(dev_loss)
            writer.add_scalar('dev loss', dev_loss_avg, (ep + 1) / config['DEV_EPOCH'])
            dev_losses.append(dev_loss_avg)
            # for statistics
            if dev_loss_avg < best_dev_loss:
                best_dev_loss = dev_loss_avg

                # save the best model based on the dev/validation set
                model.cpu()
                state = {'epoch': ep,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(config['model_save_path'],
                                               config['model_name'],
                                               'best_model'))
                model.to(device)
            logger.info("Dev Epoch:[{}]\tloss={:.3f}\taccuracy={:.3f}".format(
                            (ep + 1) // config['DEV_EPOCH'],
                            np.mean(dev_loss),
                            np.mean(dev_results)))

            model.train()

    # test immediately after training
    # for test set
    data = read_metaqa(config['test_emb_path'])
    metaqa_test = MetaQADataset(data, M_subj.size(-1))
    test_dataloader = DataLoader(dataset=metaqa_test,
                                 batch_size=128,
                                 shuffle=False,
                                 collate_fn=collate_fn)

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

    # save the final model
    model.cpu()
    state = {'epoch': ep,
             'state_dict': model.state_dict(),
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

    logger.info('Dense 1 weight')
    logger.info(str(model.dense1.weight))
    logger.info('Dense 1 bias')
    logger.info(str(model.dense1.bias))

    # how to output results. set a number? or select the top k?

if __name__ == "__main__":
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    run(config)

