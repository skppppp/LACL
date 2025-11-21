import random

from torch.utils.data import Dataset
import sys
import os
import json

curPath = os.path.abspath(os.path.dirname('__file__'))
# rootPath = os.path.split(curPath)[0]
rootPath = curPath
sys.path.append(rootPath)
import torch
from tools.utils import *
from torchtext.data import Field
# from torchtext.legacy.data import Field
from torchtext.vocab import Vectors
from random import randint, choice
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import copy
from collections import defaultdict
import itertools
import scipy.sparse as sp


class MashupDataset(Dataset):
    def __init__(self, all_api=False, api_threshold=1647):
        super().__init__()
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.name = json.load(f)
        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            self.description = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            self.category = json.load(f)
        with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
            self.used_api = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)

        ''''''
        with open(rootPath + '/data/api_category.json', 'r') as f:
            api_category = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r') as f:
            api_description = json.load(f)
        with open(rootPath + '/data/api_name.json', 'r') as f:
            api_list_all = json.load(f)

        print("api list all:{0}".format(len(api_list_all)))
        self.api_category = []
        self.api_description = []

        self.api2category = {}
        self.api2description = {}
        for api in api_list_all:
            self.api2category[api] = api_category[api_list_all.index(api)]
            self.api2description[api] = api_description[api_list_all.index(api)]

        self.API_THRESHOLD = api_threshold
        if all_api:
            with open(rootPath + '/data/used_api_list.json', 'r') as f:
                api_list = json.load(f)

            for i in api_list_all:
                # print(i)
                if len(api_list) < self.API_THRESHOLD and i not in api_list:
                    api_list.append(i)
                # else:
                #     break
            print("{0}==>{1}".format(len(api_list), self.API_THRESHOLD))
        else:
            with open(rootPath + '/data/used_api_list.json', 'r') as f:
                api_list = json.load(f)

        self.category2api = {}
        for api in api_list:
            self.api_category.append(self.api2category[api])
            self.api_description.append(self.api2description[api])

            for cat in self.api2category[api]:
                if cat not in self.category2api:
                    self.category2api[cat] = []

                self.category2api[cat].append(api)

        self.num_api = len(api_list)
        self.num_mashup = len(self.used_api)
        self.num_bundle = 0
        self.num_category = len(category_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([api_list])

        self.build_graph()
        self.build_aff_graph()
        self.build_bundle_graph()

        self.category_freq, self.api_freq = self.get_freq_matrix()
   
        self.categ = copy.deepcopy(self.category)
        # self.des = []
        self.des = copy.deepcopy(self.description)
        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)
        
        
        # self.train_idx, self.val_idx, self.test_idx, self.sample_lt = get_indices_withlt_bundle(self.ub_pairs, self.num_mashup)

        self.train_idx, self.val_idx, self.test_idx, self.sample_lt = get_indices_bundle(self.mashup_api_matrix, self.api_freq,
            threshold=4)
        
        print("train_idx:{0}, val_idx:{1}, test_idx:{2}, sample_lt_idx:{3}".format(len(self.train_idx), len(self.val_idx), len(self.test_idx), len(self.sample_lt)))

        self.train_ub_matrix = self.get_dataset_matrix(self.train_idx)
        self.val_ub_matrix = self.get_dataset_matrix(self.val_idx)
        self.test_ub_matrix = self.get_dataset_matrix(self.test_idx)

        # self.train_ub_matrix = self.get_dataset_matrix_ub(self.train_idx)
        # self.val_ub_matrix = self.get_dataset_matrix_ub(self.val_idx)
        # self.test_ub_matrix = self.get_dataset_matrix_ub(self.test_idx)

    def __len__(self):
        return len(self.name)
    
    def build_graph(self):
        self.des_lens = []
        self.category_token = []

        self.mashup_api = {}
        self.mashup_api_pair = list()
        self.mashup_api_matrix = np.zeros((self.num_mashup, self.num_api), dtype='float32')

        self.api_api_compatibility_matrix = np.zeros((self.num_api, self.num_api), dtype='float32')
        mashup_api_link = 0
        for i in range(len(self.used_api)):
            self.mashup_api.setdefault(i, {})
            self.mashup_api_pair.append([])
            api_list = self.used_api_mlb.transform([self.used_api[i]])
            api_ids = np.where(api_list[0] == 1)

            for j in api_ids[0]:
                mashup_api_link += 1
                self.mashup_api[i][j] = 1
                self.mashup_api_pair[i].append(j)
                self.mashup_api_matrix[i][j] = 1

            for i in api_ids[0]:
                for j in api_ids[0]:
                    self.api_api_compatibility_matrix[i][j] += 1
                    self.api_api_compatibility_matrix[j][i] += 1

        # self.mashup_api_matrix = random_down_sampling(self.mashup_api_matrix, 0.5)  # 采样比例为0.5
        print("Number of composition links between APIs and Mashups: {0}".format(mashup_api_link))

    def build_aff_graph(self):
        self.api_api_affinity_matrix = np.zeros((self.num_api, self.num_api), dtype='float32')        
        for cat in self.category2api.keys():
            api_list = self.used_api_mlb.transform([self.category2api[cat]])
            api_ids = np.where(api_list[0] == 1)
            for i in api_ids[0]:
                for j in api_ids[0]:
                    if i != j:
                        self.api_api_affinity_matrix[i][j] += 1
                        self.api_api_affinity_matrix[j][i] += 1
        # print(self.api_api_matrix[i][j])
        print("with category in api-api graph params...")

    def build_bundle_graph(self):
        '''
        self.bundle_indices = []
        self.bundle_item_matrix = np.zeros((0, self.num_api), dtype=int)

        for user_index in range(self.num_mashup):
            user_items = self.mashup_api_matrix[user_index]
            item_indices = np.where(user_items)[0]

            if len(item_indices) > 2:
                bundle_indice = set(itertools.combinations(item_indices, 3))
                # bundle_indice = set(itertools.combinations(item_indices, 2))
                # bundle_indice = set(itertools.combinations(item_indices, len(item_indices)))                
            else:
                # bundle_indice = set(itertools.combinations(item_indices, len(item_indices)))
                continue
            
            self.bundle_indices.extend(bundle_indice)
        
        # remove the duplicate indices
        print("Pre: Number of bundle links between APIs: {0}".format(len(self.bundle_indices)))
        self.bundle_indices = list(set(self.bundle_indices))
        self.bundle_indices = list(set(tuple(indices) for indices in self.bundle_indices))
        self.num_bundle = len(self.bundle_indices)
        print("Number of bundle links between APIs: {0}".format(len(self.bundle_indices)))

        for bundle in self.bundle_indices:
            bundle_item = np.zeros(self.num_api, dtype=int)
            bundle_item[list(bundle)] = 1
            self.bundle_item_matrix = np.vstack((self.bundle_item_matrix, bundle_item))

        # generate user_bundle_matrix through checking whether the APIs in the bundle_indices exist in the same mashup.
        self.user_bundle_matrix = np.zeros((self.num_mashup, len(self.bundle_indices)), dtype=int)
        
        print("build user_bundle_matrix...")
        
        max_bundle_size = 0
        min_bundle_size = 1000
        counter = defaultdict(int)
        for user_index in range(self.num_mashup):
            user_items = self.mashup_api_matrix[user_index]
            user_indices = np.where(user_items)[0]
            for bundle_index, bundle in enumerate(self.bundle_indices):
                # if np.all(user_items[list(bundle)]):
                #     self.user_bundle_matrix[user_index, bundle_index] = 1
                if set(bundle).issubset(user_indices):
                    self.user_bundle_matrix[user_index, bundle_index] = 1
            max_bundle_size = max(max_bundle_size, self.user_bundle_matrix[user_index].sum())
            min_bundle_size = min(min_bundle_size, self.user_bundle_matrix[user_index].sum())
            counter[self.user_bundle_matrix[user_index].sum()]+=1

        print("max_bundle_size: {0}".format(max_bundle_size))
        print("min_bundle_size: {0}".format(min_bundle_size))
        print("Number of bundle links between APIs and Mashups: {0}".format(self.user_bundle_matrix.sum()))

        sorted_dict = dict(sorted(counter.items()))
        print("bundle_range:{0}".format(sorted_dict))
        draw_counter_pic(sorted_dict, 'Sorted dict visualization of bundle')
        '''
        
        '''
        file_path = "user_bundle_matrix.txt"
        print("save user_bundle_matrix to {0}".format(file_path))
        with open(file_path, 'w') as f:
            np.savetxt(f, self.user_bundle_matrix, fmt='%d')
        print("load user-bundle matrix")
        self.user_bundle_matrix = np.loadtxt(file_path, dtype=int)
        print("load user-bundle matrix done {0}".format(self.user_bundle_matrix.shape))
        # self.ub_pairs = []
        # for i in range(self.num_mashup):
        #     user_pair = np.where(self.user_bundle_matrix[i])[0]
        #     for j in user_pair:
        #         self.ub_pairs.append((i, j))
        '''
        # 读取用户捆绑矩阵数据
        self.user_bundle_matrix = np.loadtxt('data/user_bundle_matrix.txt', delimiter=',', dtype=int)
        # 读取捆绑商品矩阵数据
        self.bundle_item_matrix = np.loadtxt('data/bundle_item_matrix.txt', delimiter=',', dtype=int)
        self.num_bundle = self.user_bundle_matrix.shape[1]
        print("Shape of user_bundle_matrix: {0}".format(self.user_bundle_matrix.shape))
        print("Shape of bundle_item_matrix: {0}".format(self.bundle_item_matrix.shape))
        print("Number of bundle links between Bundles and Items: {0}".format(self.bundle_item_matrix.sum()))
        print("Number of bundle links between Bundles and Users: {0}".format(self.user_bundle_matrix.sum()))

        self.ub_pairs = [(i, j) for i in range(self.num_mashup) 
                    for j in np.where(self.user_bundle_matrix[i])[0]]

        # print(self.ub_pairs)
        print("num of ub pairs: {0}".format(len(self.ub_pairs)))
        print("num_mashup: {0}; num_api: {1}; num_bundle: {2}".format(self.num_mashup, self.num_api, self.num_bundle))

    def get_dataset_matrix_ub(self, inds):
        data_ub_matrix = np.zeros((self.num_mashup, self.num_bundle), dtype=int)
        # get the complete (user, bundle) pairs of the train set from ub_pair
        idxs = set()
        for idx in inds:
            user, bundle = self.ub_pairs[idx]
            idxs.add(user)
            data_ub_matrix[user, bundle] = 1
        print("num of users in train set: {0}".format(len(idxs)))
        return data_ub_matrix
        
    def get_dataset_matrix(self, inds):
        data_ub_matrix = np.zeros((self.num_mashup, self.num_bundle), dtype=int)
        for user in inds:
            all_bundles = self.user_bundle_matrix[user].nonzero()[0]
            for bundle in all_bundles:
                data_ub_matrix[user, bundle] = 1
        
        return data_ub_matrix
        
    
    def get_freq_matrix(self):
        category_freq = torch.zeros(1, len(self.category_mlb.classes_))
        api_freq = torch.zeros(1, len(self.used_api_mlb.classes_))
        for category in self.category:
            category_freq += torch.tensor(self.category_mlb.transform([category])).squeeze()
        for api in self.used_api:
            api_freq += torch.tensor(self.used_api_mlb.transform([api])).squeeze()
        return category_freq, api_freq[0]
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.used_api[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])
        # category_token = self.category_token[index]

        bundle_tensor = torch.tensor(self.user_bundle_matrix[index], dtype=torch.float32).squeeze()
        bundle_target = torch.tensor(self.train_ub_matrix[index], dtype=torch.float32).squeeze()
        bundle_test_target = torch.tensor(self.test_ub_matrix[index], dtype=torch.float32).squeeze()

        return torch.tensor(index).long(), torch.tensor(
            description).long(), category_tensor, used_api_tensor, des_len, category_token, \
            bundle_tensor, bundle_target, bundle_test_target
            # self.des[index]


# def random_down_sampling(adj_matrix, sample_ratio):
#     n = adj_matrix.shape[0]  # 获取节点数
#     sample_size = int(n * sample_ratio)  # 计算采样后节点数
#     indices = np.random.choice(adj_matrix.shape[0], size=sample_size, replace=False)
#     return adj_matrix[indices][:, indices]


class ApiDataset(Dataset):
    def __init__(self, all_api=False, api_threshold=1647):
        super().__init__()
        with open(rootPath + '/data/api_name.json', 'r') as f:
            name = json.load(f)
        with open(rootPath + '/data/api_description.json', 'r') as f:
            description = json.load(f)
        with open(rootPath + '/data/api_category.json', 'r') as f:
            category = json.load(f)
        with open(rootPath + '/data/category_list.json', 'r') as f:
            category_list = json.load(f)
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            self.mashup = json.load(f)
        with open(rootPath + '/data/used_api_list.json', 'r') as f:
            used_api_list = json.load(f)

        self.API_THRESHOLD = api_threshold

        if all_api:
            '''
            self.name = name
            self.description = description
            self.category = category
            self.used_api = []
            for api in self.name:
                self.used_api.append([api])
            '''
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []

            if len(self.name) < self.API_THRESHOLD:
                for api in name:
                    if len(self.name) < self.API_THRESHOLD and api not in self.name:
                        self.name.append(api)

            print("{0}==>{1}".format(len(self.name), self.API_THRESHOLD))

            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        else:
            self.name = used_api_list
            self.description = []
            self.category = []
            self.used_api = []
            for api in self.name:
                self.description.append(description[name.index(api)])
                self.category.append(category[name.index(api)])
                self.used_api.append([api])

        self.num_category = len(category_list)
        self.num_api = len(used_api_list)
        self.category_mlb = MultiLabelBinarizer()
        self.category_mlb.fit([category_list])
        self.used_api_mlb = MultiLabelBinarizer()
        self.used_api_mlb.fit([used_api_list])
        self.des_lens = []
        self.category_token = []

        self.categ = copy.deepcopy(self.category)
        self.des = copy.deepcopy(self.description)
        # self.des = []

        for des in self.description:
            self.des_lens.append(len(des) if len(des) < 50 else 50)

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        description = self.description[index]
        category_tensor = torch.tensor(self.category_mlb.transform([self.category[index]]), dtype=torch.long).squeeze()
        used_api_tensor = torch.tensor(self.used_api_mlb.transform([self.name[index]]), dtype=torch.long).squeeze()
        des_len = torch.tensor(self.des_lens[index])
        category_token = torch.LongTensor(self.category_token[index])

        return torch.tensor(index).long(), \
               torch.tensor(description).long(), \
               category_tensor, \
               used_api_tensor, \
               des_len, category_token, \
            # self.des[index]


class BPRDataset(Dataset):
    def __init__(self, ds, sample_indices, neg_num):
        super(BPRDataset, self).__init__()
        if ds is None:
            self.ds = TextDataset()
        else:
            self.ds = ds
        self.sample_indices = sample_indices
        self.triplet = None
        self.neg_num = neg_num  # 一个正例对应需要采样的负例数量
        self.create_triplet()

    def create_triplet(self):
        pairs = []
        triplet = []
        neg_list = list(range(self.ds.num_bundle))
        for sample in self.sample_indices:
            pos_indices = self.ds.mashup_ds[sample][6].nonzero().flatten().tolist()
            for pos in pos_indices:
                if pos > self.ds.num_bundle:
                    print("pos:{0}".format(pos))
                pairs.append([sample, pos])
        for pair in pairs:
            break_point = 0
            while (True):
                ch = choice(neg_list)
                if break_point == self.neg_num:
                    break
                elif ch != pair[1]:
                    triplet.append((pair[0], pair[1], ch))
                    break_point += 1

        self.triplet = triplet

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        # mashup = self.ds.mashup_ds[sample[0]]
        # api_i = self.ds.api_ds[sample[1]]
        # api_j = self.ds.api_ds[sample[2]]
        mashup = sample[0]
        all_bundles = list([sample[1], sample[2]])
        all_bundles_feature = []
        for bundle in all_bundles:
            items = self.ds.mashup_ds.bundle_item_matrix[bundle].nonzero()[0]
            bundles = []
            for item in items:
                bundles.append(self.ds.api_ds[item][0])
            all_bundles_feature.append(bundles)
        
        ub_matrix = self.ds.mashup_ds.user_bundle_matrix[mashup]
        ui_matrix = self.ds.mashup_ds.mashup_api_matrix[mashup]

        return torch.tensor(mashup).long(), torch.tensor(all_bundles).long(), \
            torch.tensor(ub_matrix, dtype=torch.float), torch.tensor(ui_matrix, dtype=torch.float), \
            all_bundles

class BPRDataset_API(Dataset):
    def __init__(self, ds, sample_indices, neg_num):
        super(BPRDataset_API, self).__init__()
        if ds is None:
            self.ds = TextDataset()
        else:
            self.ds = ds
        self.sample_indices = sample_indices
        self.triplet = None
        self.neg_num = neg_num  # 一个正例对应需要采样的负例数量
        self.create_triplet()

    def create_triplet(self):
        pairs = []
        triplet = []
        neg_list = list(range(len(self.ds.api_ds)))
        for sample in self.sample_indices:
            pos_indices = self.ds.mashup_ds[sample][3].nonzero().flatten().tolist()
            for pos in pos_indices:
                pairs.append([sample, pos])
        for pair in pairs:
            break_point = 0
            while (True):
                ch = choice(neg_list)
                if break_point == self.neg_num:
                    break
                elif ch != pair[1]:
                    triplet.append((pair[0], pair[1], ch))
                    break_point += 1

        self.triplet = triplet

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        # mashup = self.ds.mashup_ds[sample[0]]
        # api_i = self.ds.api_ds[sample[1]]
        # api_j = self.ds.api_ds[sample[2]]
        mashup = sample[0]
        all_api = list([sample[1], sample[2]])
        return torch.tensor(mashup).long(), torch.tensor(all_api).long()

# class NNRDataset(Dataset):
#     def __init__(self, nn_num):
#         super(NNRDataset, self).__init__()
#         self.tds = TextDataset()
# 
#         # self.sample_indices = sample_indices
#         self.nn_num = nn_num  # 近邻mashup数量
#         self.sim_matrix = torch.zeros(self.nn_num, self.tds.embed_dim)
#         self.mashup_feature = torch.self.tds.embed[ds.mashup_ds.description].sum(dim=1)
#         self.sim_cal()
#
#     def sim_cal(self):
#         for i, des_list in enumerate(range(self.nn_num)):
#             tmp_ = torch.zeros(300)
#
#             m_feature[i] = torch.nn.functional.normalize(tmp_, dim=0)
#         self.sim_matrix = m_feature.mm(m_feature.t()).argsort(dim=1, descending=True)
#
#     def __len__(self):
#         return len(self.mds)
#
#     def __getitem__(self, index):
#         if torch.is_tensor(index):
#             index = index.tolist()
#         main_mashup = self.tds.mashup_ds[index]
#         nn_mashup_des = torch.zeros(self.nn_num, 50)
#         for count, i in enumerate(self.sim_matrix[index, 1:self.nn_num+1]):
#             nn_mashup_des[count] = self.tds.mashup_ds[i][1]
#
#         return main_mashup, nn_mashup_des.long()


class TextDataset:
    def __init__(self, mul=1, is_random=False):
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        USED_APIS_LENS = 1647
        Multiple = mul
        self.API_THRESHOLD = USED_APIS_LENS * Multiple
        # self.API_THRESHOLD = 23518
        self.mashup_ds = MashupDataset(all_api=True, api_threshold=self.API_THRESHOLD)
        self.api_ds = ApiDataset(all_api=True, api_threshold=self.API_THRESHOLD)
        self.max_vocab_size = 10000
        self.max_doc_len = 50
        self.vectors = Vectors(name=rootPath + '/tools/glove/glove.6B.200d.txt', cache=cache)
        self.field = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=self.max_doc_len)

        self.field.build_vocab(self.mashup_ds.description, self.api_ds.description, vectors=self.vectors, min_freq=1,
                               max_size=self.max_vocab_size)

        self.random_seed = 2020
        self.num_category = self.mashup_ds.num_category
        self.num_mashup = self.mashup_ds.num_mashup
        # self.num_api = len(self.api_ds)
        self.num_api = self.mashup_ds.num_api
        self.num_bundle = self.mashup_ds.num_bundle
        # print(self.num_api)

        self.train_idx = self.mashup_ds.train_idx
        self.test_idx = self.mashup_ds.test_idx
        self.val_idx = self.mashup_ds.val_idx
        self.sample_lt = self.mashup_ds.sample_lt

        self.vocab_size = len(self.field.vocab)
        self.embed = self.field.vocab.vectors
        self.embed_dim = self.vectors.dim
        self.des_lens = []
        self.word2id(is_random)
        self.tag2feature() 

    def word2id(self, is_random):

        counter = defaultdict(int)
        for i, des in enumerate(self.mashup_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            # self.random_mask(is_random, tokens)
            counter[len(tokens)] += 1

            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.mashup_ds.description[i] = tokens

        sorted_dict = dict(sorted(counter.items()))
        print("mashup_ds_len:{0}".format(sorted_dict))
        # draw_counter_pic(sorted_dict, 'Sorted dict visualization of mashup')

        counter = defaultdict(int)
        for i, des in enumerate(self.api_ds.description):
            tokens = [self.field.vocab.stoi[x] for x in des]
            self.random_mask(is_random, tokens)

            counter[len(tokens)] += 1

            if not tokens:
                tokens = [0]
            if len(tokens) < self.max_doc_len:
                tokens.extend([1] * (self.max_doc_len - len(tokens)))
            else:
                tokens = tokens[:self.max_doc_len]
            self.api_ds.description[i] = tokens

        sorted_dict = dict(sorted(counter.items()))
        print("api_ds_len:{0}".format(sorted_dict))
        # draw_counter_pic(sorted_dict, 'Sorted dict visualization of api')

    def random_mask(self, is_random, tokens):
        if is_random:
            n = len(tokens)
            if n < 10:
                return
            # k = random.sample(range((int)((n + 1) / 4)), 1)[0]
            # print("n:{0}".format(n))
            k = (int)(1.0 * n * 0.6)
            nums = random.sample(range(n - 1), k)
            print("n:{0}-->num:{1}".format(n, nums))
            for num in nums:
                tokens[num] = 1

    def tag2feature(self):
        for i, category in enumerate(self.mashup_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.mashup_ds.category_token.append(tokens)

        for i, category in enumerate(self.api_ds.category):
            tokens = [self.field.vocab.stoi[x] for x in tokenize(' '.join(category))]
            if not tokens:
                tokens = [0]
            if len(tokens) < 10:
                tokens.extend([1] * (10 - len(tokens)))
            else:
                tokens = tokens[:10]
            self.api_ds.category_token.append(tokens)


class F3RMDataset(Dataset):
    def __init__(self, nn_num=10):
        super(F3RMDataset, self).__init__()
        cache = '.vec_cache'
        if not os.path.exists(cache):
            os.mkdir(cache)
        self.tds = TextDataset()

        # self.sample_indices = sample_indices
        self.nn_num = nn_num  # 近邻mashup数量
        self.neighbor_mashup_des = torch.zeros(len(self.tds.mashup_ds), self.nn_num, self.tds.max_doc_len)
        self.mashup_feature = torch.nn.functional.normalize(self.tds.embed[self.tds.mashup_ds.description].sum(dim=1))
        self.sim = torch.nn.functional.normalize(torch.mm(self.mashup_feature, self.mashup_feature.t()))
        self.neighbor_mashup_index = self.sim.argsort(descending=True)[:, :self.nn_num]
        for i in range(len(self.tds.mashup_ds)):
            for j, index in enumerate(range(self.nn_num)):
                self.neighbor_mashup_des[i, j] = self.tds.mashup_ds[index][1]

    def __len__(self):
        return len(self.tds.mashup_ds)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        main_mashup = self.tds.mashup_ds[index]
        n_mashup_des = self.neighbor_mashup_des[index]
        return main_mashup, n_mashup_des.long()


class FCDataset(Dataset):
    def __init__(self, sample_indices, is_training=True):
        super(FCDataset, self).__init__()
        self.ds = TextDataset()
        self.triplet = []
        if is_training:
            self.neg_num = 14  # 一个正例对应需要采样的负例数量
            for indice in sample_indices:
                pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
                for pos in pos_indices:
                    self.triplet.append([indice, pos, 1])
                for idx in range(self.neg_num):
                    r = randint(0, 1646)
                    if r not in pos_indices:
                        self.triplet.append([indice, r, -1])
        else:
            for indice in sample_indices:
                pos_indices = self.ds.mashup_ds[indice][3].nonzero().flatten().tolist()
                for idx in range(len(self.ds.api_ds)):
                    if idx in pos_indices:
                        self.triplet.append([indice, idx, 1])
                    else:
                        self.triplet.append([indice, idx, -1])

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        sample = self.triplet[index]
        mashup = self.ds.mashup_ds[sample[0]]
        api = self.ds.api_ds[sample[1]]
        label = sample[2]
        return mashup, api, label

def saveBundle(ds):
    pass


if __name__ == '__main__':
    # mashup_ds = MashupDataset()
    # api_ds = ApiDataset()
    # ds = F3RMDataset()
    ds = TextDataset()
    saveBundle(ds)

