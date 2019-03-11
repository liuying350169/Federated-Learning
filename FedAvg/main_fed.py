#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import copy
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter

from sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_noniid_extram, cifar_noniid, cifar_noniid_extram, \
    cifar100_iid, cifar100_noniid, cifar100_noniid_extram, KWS_iid, KWS_noniid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP, CNNMnist, CNNCifar, CNNKws
from averaging import average_weights
from FedNets import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from FedNets import VGG
from FedNets import MobileNetV2
from FedNets import ShuffleNetV2


class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append((info[0], int(info[1])))
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (1, 50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # parse args
    args = args_parser()

    #write args to file
    f = open('./test.txt', 'a')
    print(args, file=f)
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

    elif args.dataset == 'fashionmnist':
        dataset_train = datasets.FashionMNIST('../data/fashion/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        dataset_test = datasets.FashionMNIST('../data/fashion/', train=False, download=True,
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
        transform_train = transforms.Compose([
            #no affect to the result
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform_train, target_transform=None, download=True)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform_test, target_transform=None, download=True)

        if args.iid == 1:
            #print("iid==1")
            dict_users = cifar_iid(dataset_train, args.num_users)
        elif args.iid == 2:
            #print("iid==2")
            dict_users = cifar_noniid_extram(dataset_train, args.num_users)
        else:
            #print("iid==0")
            dict_users = cifar_noniid(dataset_train, args.num_users)

    elif args.dataset == 'cifar100':
        exit('Error: dataset should pre-trans before use')
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, transform=transform, target_transform=None, download=True)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, transform=transform, target_transform=None, download=True)

        if args.iid==1:
            dict_users = cifar100_iid(dataset_train, args.num_users)
        elif args.iid==2:
            dict_users = cifar100_noniid_extram(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users)

    elif args.dataset == 'kws':
        dataset_train = KWSconstructor('../data/kws/index_train.txt', transform=None)
        dataset_test = KWSconstructor('../data/kws/index_test.txt', transform=None)
        if args.iid==1:
            dict_users = KWS_iid(dataset_train, args.num_users)
        else:
            dict_users = KWS_noniid(dataset_train, args.num_users)

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

    elif args.model == 'cnn' and args.dataset == 'kws':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNKws(args=args).cuda()
        else:
            net_glob = CNNKws(args=args)

    elif args.model == 'resnet18' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = ResNet18().cuda()
        else:
            net_glob = ResNet18()

    elif args.model == 'resnet34' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = ResNet34().cuda()
        else:
            net_glob = ResNet34()

    elif args.model == 'vgg16' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = VGG('VGG16').cuda()
        else:
            net_glob = VGG('VGG16')

    elif args.model == 'mobilenetv2' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MobileNetV2().cuda()
        else:
            net_glob = MobileNetV2()

    elif args.model == 'shufflenetv2' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = ShuffleNetV2(1).cuda()
        else:
            net_glob = ShuffleNetV2(1)

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
    net_glob_old = net_glob
    net_glob.train()
    net_glob_old.train()
    # copy weights
    w_glob = net_glob.state_dict()

    #net_glob is the global model
    #state_dict() is the weights of a model

    # training
    loss_train = []
    acc_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    # tqdm jin du tiao
    # allids = []
    w_locals_old = []
    for i in range(100):
        w_locals_old.append(copy.deepcopy(w_glob))
    #w_locals_old初始化
    for iter in tqdm(range(args.epochs)):
        w_locals, loss_locals = [], []
        if(args.num_users <= 10):
            m = args.num_users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            m = max(int(args.frac * args.num_users), 1)
            #m is select how many ready client to use， default is 10
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # for every select users
        # idxs_users is some numbers
        if(args.exchange == 0):
            conv1_params = 450
            conv2_params = 2400
            fc1_params = 48000
            fc2_params = 10080
            fc3_params = 840
            total_params = conv1_params+conv2_params+fc1_params+fc2_params+fc3_params
            batch_num = 4
            #x_total = [4][61770]
            x_total = [[] for i in range(total_params)]
            x_time = [[] for i in range(batch_num)]
            for i in range(batch_num):
                x_time[i] = x_total

            counter_i = 0
            for idx in tqdm(idxs_users):
                # print("user num id",idx)
                # allids.append(idx)
                # allids.sort()
                # print(allids)
                # use LocalUpdate to update weight
                # train_test_validate has [] [] []
                local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users, i=idx, tb=summary)
                # LocalUpdate initial
                net_glob_old.load_state_dict(w_locals_old[idx])


                #是否关闭同步的条件,通过round数控制，通过分歧度控制
                #1.要验证exc能否有效降低分歧度
                #2.要验证non-iid数据的分歧度上升规律
                if(iter>70):
                    ###***###
                    #不关闭同步，下一个round
                    w, loss, x = local.update_weights(net=copy.deepcopy(net_glob))
                else:
                    # 关闭同步，各自训练
                    w, loss, x = local.update_weights(net=copy.deepcopy(net_glob_old))
                ###***###
                # use global to trainde
                # w is local model's state_dict(), means the weight of local model
                # loss is the sum(epoch_loss) / len(epoch_loss)
                ### x is 4 params lists
                #print(len(x), len(x[0]), len(x[0][0]),x[0][0][0], x[0][0], x[0])
                #print(x[j][i][len(x[j][i])-2])
                #print(x[0][0][0], x[1][0][0], x[2][0][0], x[3][0][0])
                #表示x的四个batch时态的值，第二个0表示61770个参数中的第0位，最后一个0实际上没有意义，因为他就是一个【值】
                #所以最后要把61770个参数摊平到61770个list中，这61770个list叫做 x_total

                #print()
                ###version 1
                # for j in range(batch_num):
                #     for i in range(total_params):
                #         x_total[i].append(x[j][i][0])
                #         #print(i,x_total[i])
                #     x_time[j] = x_total

                #print(j,x_time[j][0][0])
                #x_total = [[] for i in range(total_params)]

                ###version 2
                for i in range(batch_num):
                    for j in range(total_params):
                        x_time[i][j].append(x[i][j][0])


                #print(x_time[0][0][0], x_time[1][0][0], x_time[2][0][0], x_time[3][0][0])
                # print(x_time[0][0], x_time[1][0][0], x_time[2][0][0], x_time[3][0][0])
                # print(len(x_time),len(x_time[0]),len(x_time[0][0]))
                # for i in range(batch_num):
                #     for j in range(total_params):
                #         x_time1[i][j]
                    #print(x[j][i][0], x[j][i][len(x[j][i])-2])
                #print(len(x_time[j][i]), x_time[j][i])

                if (counter_i % 5 == 0):

                    f_mean_std = open('./mean_std02221108.txt', 'a')
                    f_mean_var = open('./mean_var02221108.txt', 'a')
                    res_var = [[] for i in range(batch_num)]
                    res_std = [[] for i in range(batch_num)]

                    for j in range(batch_num):
                        for i in range(total_params):

                            temp_var = np.var(x_time[j][i])

                            temp_std = np.std(x_time[j][i], ddof=1)

                            res_var[j].append(temp_var)

                            res_std[j].append(temp_std)
                        #print(x_time[0][0][0],x_time[1][0][0],x_time[2][0][0],x_time[3][0][0])
                        #print(temp_var)
                            #print(len(res_var[j]))

                    #print(len(res_var[0]),res_var[0][0],len(res_var[1]),res_var[1][0],len(res_var[2]),res_var[2][0],len(res_var[3]),res_var[3][0])
                    #print(len(res_var[j]),j,res_var[j][0:10])

                    mean_var = np.mean(res_var[j])
                    print(mean_var, file=f_mean_var)
                    mean_std = np.mean(res_std[j])
                    print(mean_std, file=f_mean_std)
                    print("mean_var", mean_var)
                    #print("mean_std", mean_std)
                    res_var[j] = []
                    res_std[j] = []
                    f_mean_std.close()
                    f_mean_var.close()

                counter_i = counter_i + 1
                # w_locals is [], an empty []
                # w_locals save the local weight
                w_locals.append(copy.deepcopy(w))
                # loss_locals is [], an empty []
                # loss_locals save the loss
                loss_locals.append(copy.deepcopy(loss))
            # w_locals and loss_locals return all select idxs_users

        elif(args.exchange == 1):
            for idx in tqdm(idxs_users):
                local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users, i=idx,
                                    tb=summary)
                w, loss = local.exchange_weight(net=copy.deepcopy(net_glob))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
        # update global weights
        # average_weights all w_locals in every epoch
        w_locals_old = w_locals
        w_glob = average_weights(w_locals)
        # here can use other ways to average
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # print loss
        # calculate avg_loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        # error
        # should be iter % 10 == 0:
        # now is do it every time
        # I want to show acc every iter
        # it's truly no use
        #if args.epochs % 10 == 0:
        #print(args.epochs)
        list_acc, list_loss = [], []
        #model.eval() makes model can be test
        net_glob.eval()
        #for every user?
        #for every users is because in before every test is different, but now they are same
        #so we can use only one to test
        # if(args.alltest == 1):
        net_local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users, i=0, tb=summary)
        acc, loss = net_local.test(net=net_glob)
        #acc_avg = acc
            #loss_avg = loss
        # elif(args.alltest == 0):
        # for c in range(args.num_users):
        #     #test is not according to users, is the same
        #     #test in all 180*80 samples
        #     net_local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users, i=c, tb=summary)
        #     acc, loss = net_local.test(net=net_glob)
        list_acc.append(acc)
        list_loss.append(loss)
        acc_avg = 100. * sum(list_acc) / len(list_acc)
        f = open('./test.txt', 'a')
        print('\nTrain loss:', loss_avg)
        #print('\nTrain loss:', loss_avg,file=f)
        print("iter:{} | Train loss:{} | average acc: {:.2f}%".format(iter, loss_avg, acc_avg))
        print("iter:{} | Train loss:{} | average acc: {:.2f}%".format(iter, loss_avg, acc_avg), file=f)
        f.close()

        loss_train.append(loss_avg)
        acc_train.append(acc_avg)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('../save/loss_fed_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.num_users, args.iid, datetime.datetime.now()))

    # plot acc curve
    plt.figure()
    plt.plot(range(len(acc_train)), acc_train)
    plt.ylabel('train_acc')
    plt.savefig('../save/acc_fed_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.num_users, args.iid, datetime.datetime.now()))


    # testing
    list_acc, list_loss = [], []
    net_glob.eval()
    for c in range(args.num_users):
        net_local = LocalUpdate(args=args, dataset=dataset_train, testset=dataset_test, idxs=dict_users, i=c, tb=summary)
        acc, loss = net_local.test(net=net_glob)
        list_acc.append(acc)
        list_loss.append(loss)

    f = open('./test.txt', 'a')
    print("average acc: {:.2f}%".format(100.*sum(list_acc)/len(list_acc)))
    print("average acc: {:.2f}%".format(100. * sum(list_acc) / len(list_acc)),file=f)
    f.close()

