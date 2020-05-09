# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 13:42:11
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:28:17
import tqdm
import torch
import torch.nn as nn
from sklearn import metrics


def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().numpy(),
                                             prediction.detach().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().numpy(),
                          torch.round(prediction).detach().numpy())
    recall = metrics.recall_score(ground_truth.detach().numpy(),
                                  torch.round(prediction).detach().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().numpy(),
        torch.round(prediction).detach().numpy())

    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision))


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step

    def forward(self, pred, batch):
        loss = 0
        prediction = torch.tensor([])
        ground_truth = torch.tensor([])
        for student in range(pred.shape[0]):
            delta = batch[student][:, 0:self.num_of_questions] + batch[
                student][:, self.num_of_questions:]
            temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long)
            p = temp.gather(0, index)[0]
            a = (((batch[student][:, 0:self.num_of_questions] -
                   batch[student][:, self.num_of_questions:]).sum(1) + 1) //
                 2)[1:]
            # for i in range(len(p)):
            #     if p[i] > 0:
            #         loss = loss - (a[i]*torch.log(p[i]) + (1-a[i])*torch.log(1-p[i]))
            for i in range(len(p) - 1, -1, -1):
                if p[i] > 0:
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func):
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        pred = model(batch)
        loss, prediction, ground_truth = loss_func(pred, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, loss_func):
    ground_truth = torch.Tensor([])
    prediction = torch.Tensor([])
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        pred = model(batch)
        loss, p, a = loss_func(pred, batch)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
        # for student in range(pred.shape[0]):
        #     delta = batch[student][:, 0:self.num_of_questions] + batch[student][:,self.num_of_questions:]
        #     temp = pred[student][:C.MAX_STEP - 1].mm(delta[1:].t())
        #     index = torch.LongTensor([[i for i in range(C.MAX_STEP - 1)]])
        #     p = temp.gather(0, index)[0]
        #     a = (((batch[student][:, 0:self.num_of_questions] - batch[student][:, self.num_of_questions:]).sum(1) + 1)//2)[1:]
        #     for i in range(len(p)):
        #         if p[i] > 0:
        #             prediction = torch.cat([prediction,p[i:i+1]])
        #             ground_truth = torch.cat([ground_truth, a[i:i+1]])
    performance(ground_truth, prediction)
