import logging
import yaml

import torch
import torch.nn
from torch import optim
from torch.utils.data import DataLoader

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
    """The whole test process
    """
    logger = get_logger(config['logger_file_name']) # for logger

    # check cuda
    if config['use_cuda']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            logger.warning('Cannot find CUDA, using CPU')
    else:
        device = torch.device('cpu')

    # prepare saving environment
    save_dir = os.path.join(config['model_save_path'], config['model_name'])
    if not os.path.exists(save_dir):
        os.mkdirs(save_dir)

    # read files
    logger.info("Read files and prepare data")

    # data = read_metaqadata()
    metaqa_test = MetaQADataset(data)
    test_dataloader = DataLoader(dataset=metaqa_test, batch_size=128, shuffle=False)

        # get M_subj, M_rel, M_obj
    M_subj, M_rel, M_obj = read_KB('')
    config['N_R'] = M_rel.size(1) # add config['N_R'] here

    # train
    # assume we have M_subj, M_rel, M_obj, and dataloaders for train and dev
    logger.info("Set up model, optimizer, and criterion")
    model, _, _ = make_model(config, [M_subj, M_rel, M_obj])
    # load pre-trained data
    state = torch.load(os.path.join(save_dir, config['read_from']))
    model.load_state_dict(state['state_dict'])
    model.to(device)

    logger.info("Start testing")

    results = []

    model.eval()
    for data in test_dataloader():
        inputs, _ = data
        inputs = [x.to(device) for x in inputs]
        if 'kb_multihop' == config['task']:
            y_hat = model(*inputs)
        else:
            raise ValueError
        y_hat = torch.argmax(y_hat, dim=1)
        results.extend(y_hat.tolist())

    # convert index back to entities

if __name__ == '__main__':
    with open("../config.yml") as f:
        config = yaml.safe_load(f)
    run(config)

