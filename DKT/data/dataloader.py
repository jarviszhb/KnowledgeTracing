# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:02:25
import torch
import torch.utils.data as Data
from data.readdata import DataReader


def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('dataset/assistments/builder_train.csv',
                        'dataset/assistments/builder_test.csv', max_step,
                        num_of_questions)
    dtrain = torch.tensor(handle.getTrainData().astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(handle.getTestData().astype(float).tolist(),
                         dtype=torch.float32)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader
