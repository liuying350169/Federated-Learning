#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import datetime
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    #num_shards is 200, and idx_shard is 0-199
    idx_shard = [i for i in range(num_shards)]
    #print(idx_shard)
    #dict_users is range in num_users  default is 100   type is dictionary
    #100 ge np array
    dict_users = {i: np.array([]) for i in range(num_users)}
    #print(len(dict_users),dict_users)
    #idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
    #idxs = 1-59999
    idxs = np.arange(num_shards*num_imgs)
    #print(idxs)

    #labels is dataset's label
    #len is 60000
    #it is real label
    labels = dataset.train_labels.numpy()
    #print(len(labels))
    # for i in labels:
    #     print(labels[i])
    # sort labels
    #idxs_labels is the match between 0-59999 and the real labels of dataset
    idxs_labels = np.vstack((idxs, labels))
    #print(idxs_labels)

    #change into arrange depend on the arrange of labels
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    #print(idxs_labels)

    #finally the idxs arrange to labels
    #new ids
    #[30207  5662 55366 ... 23285 15728 11924]
    idxs = idxs_labels[0,:]
    #print(idxs)

    #print(idx_shard)
    #print(np.random.shuffle(idx_shard))
    # divide and assign
    #100 ge users
    for i in range(num_users):
        #select 2 parts from dataset for every num_users
        #use p a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        #print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        #print(idx_shard)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # print(dict_users[0])
    # x = np.random.shuffle(dict_users[0])
    # print(dict_users[0])
    #finally return each user have which 600 images ,every user have a array contain 600 ge number
    return dict_users


def mnist_noniid_extram(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_users =5
    #finally divide into 5
    #num_shards, num_imgs = 200, 300
    # num_shards is 200, and idx_shard is 0-199
    #idx_shard = [i for i in range(num_shards)]
    # dict_users is range in num_users  default is 100
    dict_users = {i: np.array([]) for i in range(num_users)}
    # idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
    idxs = np.arange(len(dataset))


    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    p1 = idxs[0:12000]
    p2 = idxs[12000:24000]
    p3 = idxs[24000:36000]
    p4 = idxs[36000:48000]
    p5 = idxs[48000:60000]
    #print(p1)

    # divide and assign
    for i in range(num_users):
        if(i==0):
            dict_users[i] = p1
        if(i==1):
            dict_users[i] = p2
        if(i==2):
            dict_users[i] = p3
        if(i==3):
            dict_users[i] = p4
        if(i==4):
            dict_users[i] = p5
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if(num_users >80):
        num_users = 80
    print(len(dataset))
    num_shards, num_imgs = 160, 300 #48000
    #num_shards is 200, and idx_shard is 0-199
    idx_shard = [i for i in range(num_shards)]
    #print(idx_shard)
    #dict_users is range in num_users  default is 100   type is dictionary
    #100 ge np array
    dict_users = {i: np.array([]) for i in range(num_users)}
    #print(len(dict_users),dict_users)
    #idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
    #idxs = 1-59999
    idxs = np.arange(num_shards*num_imgs)
    print(idxs)

    #labels is dataset's label
    #len is 60000
    #it is real label
    labels = dataset.train_labels
    print(len(labels))
    labels = labels[0:48000]
    print(len(labels))

    # for i in labels:
    #     print(labels[i])
    # sort labels
    #idxs_labels is the match between 0-59999 and the real labels of dataset
    idxs_labels = np.vstack((idxs, labels))
    print(idxs_labels)
    #print(idxs_labels)

    #change into arrange depend on the arrange of labels
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    print(idxs_labels)
    #print(idxs_labels)

    #finally the idxs arrange to labels
    #new ids
    #[30207  5662 55366 ... 23285 15728 11924]
    idxs = idxs_labels[0,:]
    print(idxs)
    #print(idx_shard)
    #print(np.random.shuffle(idx_shard))
    # divide and assign
    #100 ge users
    for i in range(num_users):
        #select 2 parts from dataset for every num_users
        #use p a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        #print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        #print(idx_shard)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # print(dict_users[0])
    # x = np.random.shuffle(dict_users[0])
    # print(dict_users[0])
    #finally return each user have which 600 images ,every user have a array contain 600 ge number
    return dict_users


def cifar_noniid_extram(dataset, num_users):
    """
    Sample non-I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = 5
    # num_shards, num_imgs = 160, 300  # 48000
    # num_shards is 200, and idx_shard is 0-199
    #idx_shard = [i for i in range(num_users)]
    # print(idx_shard)
    # dict_users is range in num_users  default is 100   type is dictionary
    # 100 ge np array
    dict_users = {i: np.array([]) for i in range(num_users)}
    # print(len(dict_users),dict_users)
    # idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
    # idxs = 1-59999
    idxs = np.arange(len(dataset))
    print(idxs)

    # labels is dataset's label
    # len is 60000
    # it is real label
    labels = dataset.train_labels
    print(len(labels))

    # for i in labels:
    #     print(labels[i])
    # sort labels
    # idxs_labels is the match between 0-59999 and the real labels of dataset
    idxs_labels = np.vstack((idxs, labels))
    print(idxs_labels)
    # print(idxs_labels)

    # change into arrange depend on the arrange of labels
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    print(idxs_labels)
    # print(idxs_labels)

    # finally the idxs arrange to labels
    # new ids
    # [30207  5662 55366 ... 23285 15728 11924]
    idxs = idxs_labels[0, :]
    p1 = idxs[0:10000]
    p2 = idxs[10000:20000]
    p3 = idxs[20000:30000]
    p4 = idxs[30000:40000]
    p5 = idxs[40000:50000]
    #print(p1)

    # divide and assign
    for i in range(num_users):
        if(i==0):
            dict_users[i] = p1
        if(i==1):
            dict_users[i] = p2
        if(i==2):
            dict_users[i] = p3
        if(i==3):
            dict_users[i] = p4
        if(i==4):
            dict_users[i] = p5
    return dict_users

def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    # print(num_items)
    # print(len(dataset.train_data))
    # print(dataset.train_labels)

    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    #print(dict_users[0])
    return dict_users

def cifar100_noniid(dataset, num_users):
    """
    Sample non-I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    if(num_users >80):
        num_users = 80
    print(len(dataset))
    num_shards, num_imgs = 160, 300 #48000
    #num_shards is 200, and idx_shard is 0-199
    idx_shard = [i for i in range(num_shards)]
    #print(idx_shard)
    #dict_users is range in num_users  default is 100   type is dictionary
    #100 ge np array
    dict_users = {i: np.array([]) for i in range(num_users)}
    #print(len(dict_users),dict_users)
    #idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
    #idxs = 1-59999
    idxs = np.arange(num_shards*num_imgs)

    #labels is dataset's label
    #len is 60000
    #it is real label
    labels = dataset.train_labels
    print(len(labels))
    labels = labels[0:48000]
    print(len(labels))

    # for i in labels:
    #     print(labels[i])
    # sort labels
    #idxs_labels is the match between 0-59999 and the real labels of dataset
    idxs_labels = np.vstack((idxs, labels))
    print(idxs_labels)
    #print(idxs_labels)

    #change into arrange depend on the arrange of labels
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    print(idxs_labels)
    #print(idxs_labels)

    #finally the idxs arrange to labels
    #new ids
    #[30207  5662 55366 ... 23285 15728 11924]
    idxs = idxs_labels[0,:]
    print(idxs)
    #print(idx_shard)
    #print(np.random.shuffle(idx_shard))
    # divide and assign
    #100 ge users
    for i in range(num_users):
        #select 2 parts from dataset for every num_users
        #use p a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        #print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        #print(idx_shard)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # print(dict_users[0])
    # x = np.random.shuffle(dict_users[0])
    # print(dict_users[0])
    #finally return each user have which 600 images ,every user have a array contain 600 ge number
    return dict_users

def cifar100_noniid_extram(dataset, num_users):
    """
    Sample non-I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_users = 50
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels

    idxs_labels = np.vstack((idxs, labels))
    print(idxs_labels)
    #print(idxs_labels)

    #change into arrange depend on the arrange of labels
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    print(idxs_labels)

    idxs = idxs_labels[0,:]
    print(idxs)
    #print(idx_shard)
    #print(np.random.shuffle(idx_shard))
    # divide and assign
    #100 ge users
    begin = 0
    end = begin + 1000
    for i in range(num_users):
        dict_users[i] = idxs[begin:end]
        begin = begin+1000
        end = end + 1000
        #print(begin)
        #print(dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train_cifar = datasets.CIFAR10('../data/cifar', train=True, transform=transform, target_transform=None,
                                     download=True)
    dataset_train_cifar100 = datasets.CIFAR100('../data/cifar100', train=True, transform=transform, target_transform=None,
                                     download=True)

    print(datetime.datetime.now())
    num = 100
    #d = mnist_noniid(dataset_train, num)
    #e = mnist_noniid_extram(dataset_train, num)
    #ee = cifar_noniid_extram(dataset_train_cifar,num)
    #c100 = cifar100_iid(dataset_train_cifar100,num)
    #c100_noniid = cifar100_noniid(dataset_train_cifar100, num)
    c100_noniid = cifar100_noniid_extram(dataset_train_cifar100, num)
    print("good")