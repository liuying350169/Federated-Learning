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
        #loss func is CrossEntropyLoss or NLLoss
        #CrossEntropyLoss is used in non-softmax
        #NLLoss is used in softmax
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.testset = testset
        self.idxs = idxs
        self.i = i
        self.tb = tb
        self.ldr_train, self.ldr_val, self.ldr_test = self.train_val_test(dataset, testset, list(idxs[i]))


    def train_val_test(self, dataset, testset, idxs):
        np.random.shuffle(idxs)
        #train all trainset
        idxs_train = idxs

        idxs_val = np.arange(100)

        idxs_test = np.arange(len(testset))

        train = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        val = DataLoader(DatasetSplit(testset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=True)
        test = DataLoader(DatasetSplit(testset, idxs_test), batch_size=int(len(idxs_test)/100), shuffle=True)

        return train, val, test


    def update_weights(self, net):
        net.train()
        # train and update
        #print("optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5, weight_decay=5e-4)")
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        # conv1_params = 450
        # conv2_params = 2400
        # fc1_params = 48000
        # fc2_params = 10080
        # fc3_params = 840
        # batch_num = 4
        # x = [[] for i in range(batch_num)]
        # x_conv1 = [[] for i in range(conv1_params)]
        # x_conv2 = [[] for i in range(conv2_params)]
        # x_fc1 = [[] for i in range(fc1_params)]
        # x_fc2 = [[] for i in range(fc2_params)]
        # x_fc3 = [[] for i in range(fc3_params)]
        # counter_i = 0



        for iter in range(self.args.local_ep):
            #print(iter)
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



                #for every batch, collect the params of each layers
                # params = net.state_dict()
                # a_conv1 = params['conv1.weight'].cpu().numpy().flatten()
                # for i in range(conv1_params):
                #     x_conv1[i].append(a_conv1[i])
                #
                # a_conv2 = params['conv2.weight'].cpu().numpy().flatten()
                # for i in range(conv2_params):
                #     x_conv2[i].append(a_conv2[i])
                #
                # a_fc1 = params['fc1.weight'].cpu().numpy().flatten()
                # for i in range(fc1_params):
                #     x_fc1[i].append(a_fc1[i])
                #
                # a_fc2 = params['fc2.weight'].cpu().numpy().flatten()
                # for i in range(fc2_params):
                #     x_fc2[i].append(a_fc2[i])
                #
                # a_fc3 = params['fc3.weight'].cpu().numpy().flatten()
                # for i in range(fc3_params):
                #     x_fc3[i].append(a_fc3[i])

                # x[counter_i] = np.concatenate((x_conv1, x_conv2, x_fc1, x_fc2, x_fc3), axis=0)
                # x_conv1 = [[] for i in range(conv1_params)]
                # x_conv2 = [[] for i in range(conv2_params)]
                # x_fc1 = [[] for i in range(fc1_params)]
                # x_fc2 = [[] for i in range(fc2_params)]
                # x_fc3 = [[] for i in range(fc3_params)]
                # counter_i = (counter_i+1) % batch_num

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # for i in range(120):
        #     print(len(x), len(x[0]),len(x[i][0]), x[i][0][len(x[i][0])-2])
        #print(x[0][0][0],x[1][0][0],x[2][0][0],x[3][0][0])
        #return net.state_dict(), sum(epoch_loss) / len(epoch_loss),x
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def exchange_weight(self, net):
        net.train()
        #print("optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)")
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        epoch_loss = []
        #exchange trainset in how many clients
        idxs = self.idxs
        subset_clients = int(self.args.num_users/self.args.groups)

        if(self.args.local_ex!=subset_clients):
            exit('Error: self.args.local_ex!=subset_clients')

        for iter in range(self.args.local_ex):
            batch_loss = []
            #for every sample in trainset to prob, and calcu the loss then optimizer
            #we should change the trainset every time

            #i = (self.i + iter) % self.args.num_users
            if (self.args.groups ==2):
                group = int(self.i / 10)
                if(group > 4):
                    i = int(50 + iter)
                else:
                    i = int(0 + iter)

            if(self.args.groups == 20):
                group = int(self.i / 10)
                group_2 = int(self.i % 10)
                if(group_2 > 4):
                    i = int(group * 10 + 5 + iter)
                else:
                    i = int(group * 10 + 0 + iter)

            if(self.args.groups == 10):
                group = int(self.i / 10)
                i = int(group * 10 + iter)

            if(self.args.groups == 5):
                group = int(self.i / 10)
                if(group % 2 == 0): #even
                    i = int(group * 10 + iter)
                if(group % 2 == 1):  # odd
                    i = int((group-1)*10 +iter)

            ldr_train, ldr_val, ldr_test = self.train_val_test(self.dataset, self.testset, list(idxs[i]))
            for iter_ep in range(self.args.local_ep):
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
            list_acc.append(acc)
            list_loss.append(loss)

        avg_acc = sum(list_acc)/len(list_acc)
        avg_loss = loss
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