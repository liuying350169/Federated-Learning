#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn import metrics
from tensorboardX import SummaryWriter
from options import args_parser
from sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_extram, cifar_noniid, cifar_noniid_extram, cifar100_iid, cifar100_noniid, cifar100_noniid_extram

from torchvision import datasets, transforms


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        #error liuying
        #only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got numpy.float64)
        #print("item",item)
        #print("self.idxs[item]",self.idxs[item])
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset, testset, idxs, i, tb):
        #idxs is one selected user's imgs list
        #idxs change to the all idxs_user and i means which one
        self.args = args
        self.loss_func = nn.NLLLoss()
        self.dataset = dataset
        self.testset = testset
        self.idxs = idxs
        self.i = i
        self.tb = tb
        self.ldr_train, self.ldr_val, self.ldr_test = self.train_val_test(dataset, testset, list(idxs[i]))


    def train_val_test(self, dataset, testset, idxs):
        #total = len(idxs)
        # if(self.args.alltest == 1):
        #     np.random.shuffle(idxs)
        #     idxs_train = idxs[0:600]
        #     #print(idxs_train)
        #     idxs_val = np.arange(3000)
        #     #print(idxs_val)
        #     idxs_test = np.arange(10000)
        #     #print(idxs_test)
        #     train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        #     val = DataLoader(DatasetSplit(testset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=True)
        #     test = DataLoader(DatasetSplit(testset, idxs_test), batch_size=int(len(idxs_test)/1000), shuffle=True)
        #     # for batch_idx, (images, labels) in enumerate(test):
        #     #     print(batch_idx)
        #     #     print(labels)
        # if(self.args.alltest == 0):
        np.random.shuffle(idxs)
        #total = np.arange(10000)
        #np.random.shuffle(total)
        idxs_train = idxs[0:420]
        idxs_val = idxs[420:480]

        idxs_test = idxs[480:600]
        #print(idxs_test)
        # idxs_test1 = list(total[120*self.i:120*(self.i+1)])
        # print(idxs_test1)
        train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        val = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=True)
        test = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=True)

        return train, val, test


    def update_weights(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            #enumerate is meijv, means for everyone
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def exchange_weight(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        #exchange trainset in how many clients
        idxs = self.idxs
        for iter in range(self.args.local_ep):
            batch_loss = []
            #for every sample in trainset to prob, and calcu the loss then optimizer
            #we should change the trainset every time
            i = (self.i + iter) % self.args.num_users
            ldr_train, ldr_val, ldr_test = self.train_val_test(self.dataset, self.testset, list(idxs[i]))
            for batch_idx, (images, labels) in enumerate(ldr_train):
                if self.args.gpu != -1:
                    images, labels = images.cuda(), labels.cuda()
                images, labels = autograd.Variable(images), autograd.Variable(labels)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.gpu != -1:
                    loss = loss.cpu()
                self.tb.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test(self, net):
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, weight_decay=2)
        # for iter in range(self.args.local_ep):
        #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
        #         if self.args.gpu != -1:
        #             images, labels = images.cuda(), labels.cuda()
        #         images, labels = autograd.Variable(images), autograd.Variable(labels)
        #         net.zero_grad()
        #         log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         loss.backward()
        #         optimizer.step()
        f_prob = open('./probs.txt', 'a')
        print("new round###self.i:{}".format(self.i), file=f_prob)
        list_acc, list_loss = [], []
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            #after dataloader ldr_test there are three parts in ldr_test
            #batch_idx is the no. of which batch
            #(images, labels) are the pair of a sample
            if self.args.gpu != -1:
                images, labels = images.cuda(), labels.cuda()

            images, labels = autograd.Variable(images), autograd.Variable(labels)
            log_probs = net(images)
            loss = self.loss_func(log_probs, labels)
            #calcu the loss
            if self.args.gpu != -1:
                loss = loss.cpu()
                log_probs = log_probs.cpu()
                labels = labels.cpu()
            #calcu acc and loss for every batch
            y_pred = np.argmax(log_probs.data, axis=1)
            acc = metrics.accuracy_score(y_true=labels.data, y_pred=y_pred)
            loss = loss.item()
            print("batch_idx:{}|\ny_pred:{}|\nlabels:{}|\nacc:{}".format(batch_idx, y_pred, labels, acc), file=f_prob)
            list_acc.append(acc)
            list_loss.append(loss)

        avg_acc = sum(list_acc)/len(list_acc)
        avg_loss = loss
        print("####avgacc:{}".format(avg_acc),file=f_prob)
        f_prob.close()
        return  avg_acc, avg_loss

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train_cifar = datasets.CIFAR10('../data/cifar', train=True, transform=transform, target_transform=None,
                                     download=True)
    dataset_test_cifar = datasets.CIFAR10('../data/cifar', train=False, transform=transform, target_transform=None,
                                     download=True)
    num = 80

    c = cifar_iid(dataset_train_cifar, num)

    summary = SummaryWriter('local')

    args = args_parser()
    print("args", args.exchange )

    m = 10

    #idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    u = LocalUpdate(args=args, dataset=dataset_train_cifar, testset=dataset_test_cifar, idxs=c, i=0, tb=summary)

    print("good")