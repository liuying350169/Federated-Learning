#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os
import copy
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter

from sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_extram, cifar_noniid, cifar_noniid_extram, cifar100_iid, cifar100_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP, CNNMnist, CNNCifar
from averaging import average_weights


def test(net_g, data_loader, args):
    # testing
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.nll_loss(log_probs, target, size_average=False).item() # sum up batch loss
        y_pred = log_probs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    f = open('./test.txt', 'a')
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)),file=f)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    f.close()
    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()

    #write args to file
    f = open('./test.txt', 'a')
    print(args,file=f)
    f.close()
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        # sample users
        if args.iid == 1:
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.iid == 2:
            #return 5 shards and each shard has 12000 imgs
            dict_users = mnist_noniid_extram(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform, target_transform=None, download=True)

        if args.iid == 1:
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.iid == 2:
            dict_users = cifar_noniid_extram(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar100':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, transform=transform, target_transform=None, download=True)

        if args.iid:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape



    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)

    elif args.model == 'cnn' and args.dataset == 'cifar100':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).cuda()
        else:
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    for iter in tqdm(range(args.epochs)):
        w_locals, loss_locals = [], []
        if(args.num_users == 5):
            m = 5
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            m = max(int(args.frac * args.num_users), 1)
            #m is select how many ready client to use， default is 10
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        #for every select users
        for idx in idxs_users:
            #use LocalUpdate to update weight
            #train_test_validate has [] [] []
            local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users[idx], tb=summary)
            w, loss = local.update_weights(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = average_weights(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if args.epochs % 10 == 0:
            list_acc, list_loss = [], []
            net_glob.eval()
            for c in tqdm(range(args.num_users)):
                net_local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users[c], tb=summary)
                acc, loss = net_local.test(net=net_glob)
                list_acc.append(acc)
                list_loss.append(loss)
            f = open('./test.txt', 'a')
            print('\nTrain loss:', loss_avg)
            #print('\nTrain loss:', loss_avg,file=f)
            print("iter:{} | Train loss:{} | average acc: {:.2f}%".format(iter,loss_avg,100. * sum(list_acc) / len(list_acc)))
            print("iter:{} | Train loss:{} | average acc: {:.2f}%".format(iter,loss_avg,100. * sum(list_acc) / len(list_acc)), file=f)
            f.close()
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('../save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    list_acc, list_loss = [], []
    net_glob.eval()
    for c in tqdm(range(args.num_users)):
        net_local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users[c], tb=summary)
        acc, loss = net_local.test(net=net_glob)
        list_acc.append(acc)
        list_loss.append(loss)

    f = open('./test.txt', 'a')
    print("average acc: {:.2f}%".format(100.*sum(list_acc)/len(list_acc)))
    print("average acc: {:.2f}%".format(100. * sum(list_acc) / len(list_acc)),file=f)
    f.close()

