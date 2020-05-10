# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 21:50:46
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:20:09
"""
Usage:
    run.py [options]

Options:
    --step=<int>                        max step [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 10]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 10]
    --layers=<int>                      layers of rnn [default: 1]
"""

import os
import random
import logging
import torch

import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
from DKT.model.RNNModel import RNNModel
from DKT.data.dataloader import getDataLoader
from DKT.evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    handler = logging.FileHandler(
        f'DKT/.log/{date.year}_{date.month}_{date.day}_result.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    args = docopt(__doc__)
    step = int(args['--step'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])

    logger.info(list(args.items()))

    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    trainLoader, testLoade = getDataLoader(bs, questions, step)
    rnn = RNNModel(questions * 2, hidden, layers, questions, device)
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, step, device)

    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        rnn, optimizer = eval.train_epoch(rnn, trainLoader, optimizer,
                                          loss_func, device)
        logger.info(f'epoch {epoch}')
        eval.test_epoch(rnn, testLoade, loss_func, device)


if __name__ == '__main__':
    main()
