import logging
import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.data import DataLoader
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

    # prepare saving environment
    save_path = os.path.join(config['model_save_path'], config['model_name'])
    if not os.path.exists(save_path):
        os.mkdirs(save_path)

    # read files
    logger.info("Read files and prepare data")

    # for train set
    # data = read_metaqadata()
    metaqa_train = MetaQADataset(data)
    train_dataloader = DataLoader(dataset=metaqa_train, batch_size=128, shuffle=True)

    # for dev/validation set
    # data = read_metaqadata()
    metaqa_dev = MetaQADataset(data)
    dev_dataloader = DataLoader(dataset=metaqa_dev, batch_size=128, shuffle=True)
    # get M_subj, M_rel, M_obj
    M_subj, M_rel, M_obj = read_KB('')
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, optimizer, criterion = make_model(config, [M_subj, M_rel, M_obj])

    train_losses = []
    dev_losses = []
    best_train_loss = 1e9
    best_dev_loss = 1e9

    logger.info("Start training")
    
    model.train()
    for ep in range(config['MAX_TRAIN_EPOCH']):
        for batch_idx, data in enumerate(train_dataloader):
            # forward
            inputs, y = data # inputs: x, q, n_hop
            if 'kb_multihop' == config['task']:
                x = model(*inputs)
            else:
                raise ValueError

            # calculation loss
            loss = criterion(x, y)
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
                        ep + 1, config['MAX_TRAIN_EPOCH'], torch.mean(train_losses)))

        # validation/development set
        if (ep + 1) % config['DEV_EPOCH'] == 0:
            model.eval()
            dev_loss = []
            for batch_idx, data in enumerate(dev_dataloader):
                inputs, y = data # inputs: x, q, n_hop
                for i in range(n_hop):
                    y_hat = model(*inputs)
                loss = criterion(y_hat, y)
                loss_value = loss.item()
                dev_loss.append(loss_value)
                writer.add_scalar('dev loss raw', loss_value,
                                    (((ep + 1) / config['DEV_EPOCH'] - 1) *
                                    len(dev_dataloader) + batch_idx))

            dev_loss_avg = torch.mean(dev_loss)
            writer.add_scalar('dev loss', dev_loss_avg, (ep + 1) / config['DEV_EPOCH'])
            dev_losses.append(dev_loss_avg)
            # for statistics
            if dev_loss_avg < best_dev_loss:
                best_dev_loss = loss_dev_loss_avg

                # save the best model based on the dev/validation set
                state = {'epoch': ep,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state, os.path.join(config['model_save_path'],
                                               config['model_name'],
                                               'best_model')

            logger.info("Dev Epoch:[{}]\t loss={:.3f}".format(
                            (ep + 1) / config['DEV_EPOCH'], torch.mean(dev_loss)))

    # save the final model
    state = {'epoch': ep,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(config['model_save_path'],
                                   config['model_name'],
                                   'final_model'))
    # save the losses in text to prevent error from tensorboard
    save_dir = os.path.join(config['model_save_path'], config['model_name'])
    with open(os.path.join(save_dir, 'train_loss.txt'), 'w') as f:
        train_losses_str = ' '.join([str(loss) for loss in train_losses])
        f.write(train_losses_str)
    with open(os.path.join(save_dir, 'dev_loss.txt'), 'w') as f:
        dev_losses_str = ' '.join([str(loss) for loss in dev_losses])
        f.write(dev_losses_str)

    logger.info("Finish training")

    # how to output results. set a number? or select the top k?

if __name__ == "__main__":
    with open("../config.yml") as f:
        config = yaml.safe_load(f)
    run(config)
