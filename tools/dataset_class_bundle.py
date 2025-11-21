#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader

curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
rootPath = curPath
sys.path.append(rootPath)

def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Sum of interactions', X.sum(1).sum(0).item())
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        # self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample
        print("len of u_b_pairs: ", len(self.u_b_pairs))


    def __getitem__(self, index):
        # conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)

class ItemsTrainDataset(Dataset):
    def __init__(self, u_i_pairs, u_i_graph, num_items, neg_sample=1):
        self.u_i_graph = u_i_graph
        self.u_i_pairs = u_i_pairs
        # self.u_i_pairs = []
        # for u, b in u_b_pairs:
        #     for i in b_i_graph[b].nonzero()[1]:
        #         self.u_i_pairs.append([u, i])
        print("len of u_i_pairs: ", len(self.u_i_pairs))
        self.num_items = num_items
        self.neg_sample = neg_sample


    def __getitem__(self, index):
        # conf = self.conf
        user_i, pos_items = self.u_i_pairs[index]
        all_items = [pos_items]

        while True:
            i = np.random.randint(self.num_items)
            if self.u_i_graph[user_i, i] == 0 and not i in all_items:                                                          
                all_items.append(i)                                                                                                   
                if len(all_items) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_i]), torch.LongTensor(all_items)


    def __len__(self):
        return len(self.u_i_pairs)

class BundleTrainMatrixDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, num_bundles):
        # self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles

        print("BundleTrainMatrixDataset: u_b_graph shape: ", self.u_b_graph.shape)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        return index, u_b_grd


    def __len__(self):
        return self.u_b_graph.shape[0]

class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class BundleDatasets():
    def __init__(self, data_name, batch_size_train=2048, batch_size_test=1, neg_num=1, matrix=False):
        self.path = rootPath + '/data/datasets/'
        self.name = data_name
        # batch_size_train = conf['batch_size_train']
        # batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_i_pairs_train = self.get_train_ui(u_i_graph)

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None
        self.bundle_train_data = None
        
        if matrix:
            self.bundle_train_data = BundleTrainMatrixDataset(u_b_pairs_train, u_b_graph_train, self.num_bundles)
        else:
            self.bundle_train_data = BundleTrainDataset(u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_num)
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)
        self.items_train_data = ItemsTrainDataset(u_i_pairs_train, u_i_graph, self.num_items, neg_num)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False)
        self.items_train_loader = DataLoader(self.items_train_data, batch_size=batch_size_train, shuffle=True, drop_last=True)

    def get_train_ui(self, u_i_graph):
        u_i_pairs = []
        matrix = u_i_graph.toarray()
        ratio = 0.6
        # 计算每个用户的项数量
        num_items_per_user = np.sum(matrix, axis=1)

        # 计算每个用户需要提取的项数量
        num_items_extracted_per_user = np.round(num_items_per_user * ratio).astype(int)
        for user in range(self.num_users):
            items = matrix[user, :]
            item_indices = np.where(items)[0]
            selected_indices = np.random.choice(item_indices, size=num_items_extracted_per_user[user], replace=False)

            # 构建用户-项对
            u_i_pairs.extend([(user, item) for item in selected_indices])

        print("total train user-item: {0}".format(len(u_i_pairs))) 
        return u_i_pairs

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_aux_graph(self, u_i_graph, b_i_graph, conf):
        u_b_from_i = u_i_graph @ b_i_graph.T
        u_b_from_i = u_b_from_i.todense()
        bn1_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        u_b_for_neg_sample = np.argsort(u_b_from_i, axis=1)[:, bn1_window[0]:bn1_window[1]]

        b_b_from_i = b_i_graph @ b_i_graph.T
        b_b_from_i = b_b_from_i.todense()
        bn2_window = [int(i*self.num_bundles) for i in conf['hard_window']]
        b_b_for_neg_sample = np.argsort(b_b_from_i, axis=1)[:, bn2_window[0]:bn2_window[1]]

        return u_b_for_neg_sample, b_b_for_neg_sample


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph


if __name__ == "__main__":
    ds = BundleDatasets("iFashion")
