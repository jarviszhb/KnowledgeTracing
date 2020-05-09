# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 21:50:46
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:19:10
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

import random
import torch

import torch.optim as optim
import numpy as np

from docopt import docopt
from model.RNNModel import RNNModel
from data.dataloader import getDataLoader
from evaluation import eval


def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = docopt(__doc__)
    step = int(args['--step'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    # cuda = int(args['--cuda'])
    hidden = int(args['--hidden'])
    layers = int(args['--layers'])

    setup_seed(seed)
    trainLoader, testLoade = getDataLoader(bs, questions, step)

    rnn = RNNModel(questions * 2, hidden, layers, questions)
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    loss_func = eval.lossFunc(questions, step)

    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        rnn, optimizer = eval.train_epoch(rnn, trainLoader, optimizer,
                                          loss_func)
        eval.test_epoch(rnn, testLoade, loss_func)


if __name__ == '__main__':
    main()
