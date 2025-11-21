# -*- conding: utf-8 -*-
"""
@File   : metric.py
@Time   : 2021/1/9
@Author : yhduan
@Desc   : None
"""
import math

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
from scipy import stats
import torch.nn.functional as F

def metric(batch_target, batch_pred, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])
    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size


def metric2(target, pred, top_k_list):
    target = target.nonzero().squeeze().tolist()

    if isinstance(target, int):
        target = [target]

    if len(target) == 0:
        return 0, 0, 0, 0

    pred = pred.argsort(descending=True).tolist()
    if isinstance(pred, int):
        target = [pred]
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    for i, k in enumerate(top_k_list):
        _ndcg[i] += ndcg(target, pred[:k])
        _recall[i] += recall(target, pred[:k])
        _map[i] += ap(target, pred[:k])
        _pre[i] += precision(target, pred[:k])

    return _ndcg, _recall, _map, _pre


def metric3(batch_target, batch_pred, api_freq, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    _div = np.zeros(len(top_k_list))
    _div_pred = np.zeros(len(top_k_list))
    _div_tar = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

            div_rate, div_pred, div_target = giniIndex(_target, _pred[:k], api_freq)

            _div[i] += div_rate
            _div_pred[i] += div_pred
            _div_tar[i] += div_target

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size, _div / batch_size, \
           [_div_pred / batch_size, _div_tar / batch_size]


def metric4(batch_target, batch_pred, api_tag_embed, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    _div = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

            _div[i] += diversity(_target, _pred[:k], api_tag_embed)

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size, _div / batch_size


def metric5(batch_target, batch_pred, api_tag_embed, popular_items, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    _div = np.zeros(len(top_k_list))
    _pb = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

            _div[i] += diversity(_target, _pred[:k], api_tag_embed)
            _pb[i] += popular_bias(_pred[:k], popular_items)

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size, _div / batch_size, _pb / batch_size



def metric_bundle(batch_target, batch_pred, api_tag_embed, popular_items, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    _pb = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

            _pb[i] += popular_bias(_pred[:k], popular_items)

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size, _pb / batch_size


def metric_bundle_pw(batch_target, batch_pred, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))

    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size


def metric_bundle_normal(batch_target, batch_pred, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))
    
    for _target, _pred in zip(batch_target, batch_pred):
        _target = _target.nonzero().squeeze().tolist()
        if isinstance(_target, int):
            _target = [_target]
        if len(_target) == 0:
            batch_size -= 1
            continue
        _pred = _pred.argsort(descending=True).tolist()
        for i, k in enumerate(top_k_list):
            _ndcg[i] += ndcg(_target, _pred[:k])
            _recall[i] += recall(_target, _pred[:k])
            _map[i] += ap(_target, _pred[:k])
            _pre[i] += precision(_target, _pred[:k])

    # _auc / batch_size
    return _ndcg / batch_size, _recall / batch_size, _map / batch_size, _pre / batch_size

_is_hit_cache = {}

def get_is_hit(scores, ground_truth, topk):
    global _is_hit_cache
    cacheid = (id(scores), id(ground_truth))
    if topk in _is_hit_cache and _is_hit_cache[topk]['id'] == cacheid:
        return _is_hit_cache[topk]['is_hit']
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1),
                              col_indice.view(-1)].view(-1, topk)
        _is_hit_cache[topk] = {'id': cacheid, 'is_hit': is_hit}
        return is_hit

def metric_bundle_normal_dp(batch_target, batch_pred, top_k_list):
    """

    :param batch_target: [batch_size, num_label]
    :param batch_pred: [batch_size, num_label]
    :param top_k_list: [k]
    :return: ndcg, recall, map, precision
    """
    # batch_size = batch_target.size(0)
    _ndcg = np.zeros(len(top_k_list))
    _recall = np.zeros(len(top_k_list))
    _map = np.zeros(len(top_k_list))
    _pre = np.zeros(len(top_k_list))
    
    for i, k in enumerate(top_k_list):
        _, col_indice = torch.topk(batch_pred, k)
        row_indice = torch.zeros_like(col_indice) + torch.arange(batch_pred.shape[0], device=batch_pred.device, dtype=torch.long).view(-1, 1)
        is_hit = batch_target[row_indice.view(-1), col_indice.view(-1)].view(-1, k)
        _ndcg[i] += get_ndcg(batch_pred, batch_target, is_hit, k)
        _recall[i] += get_recall(batch_pred, batch_target, is_hit, k)
        # _map[i] += get_map(batch_pred, batch_target, is_hit, k)
        _pre[i] += get_pre(batch_pred, batch_target, is_hit, k)

        # _ndcg[i] += ndcg(_target, _pred[:k])
        # _recall[i] += recall(_target, _pred[:k])
        # _map[i] += ap(_target, _pred[:k])
        # _pre[i] += precision(_target, _pred[:k])


    # _auc / batch_size
    return _ndcg, _recall, _pre

def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return nomina / float(denorm)

# def get_map(pred, grd, is_hit, topk):
#     epsilon = 1e-8
#     hit_cnt = is_hit.sum(dim=1)
#     num_pos = grd.sum(dim=1)

#     # remove those test cases who don't have any positive items
#     denorm = pred.shape[0] - (num_pos == 0).sum().item()
#     nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

#     return nomina / float(denorm)

def get_pre(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    '''
    epsilon = 1e-8
    hit_set = list(set(target) & set(pred))
    return len(hit_set) / float(len(pred) + epsilon)
    '''

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    # torch.topk(pred, topk)[0].sum(dim=1)
    # topk_score = torch.topk(pred, topk)[0].sum(dim=1)
    topk_score = topk
    nomina = (hit_cnt/(topk_score+epsilon)).sum().item()

    return nomina / float(denorm)

def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return nomina / float(denorm)

def ndcg(target, pred):
    dcg = 0
    c = 0
    for i in range(1, len(pred) + 1):
        rel = 0
        if pred[i - 1] in target:
            rel = 1
            c += 1
        dcg += (np.power(2, rel) - 1) / np.log2(i + 1)
    if c == 0:
        return 0
    idcg = 0
    for i in range(1, c + 1):
        idcg += (1 / np.log2(i + 1))
    return dcg / idcg


def ap(target, pred):
    p_at_k = np.zeros(len(pred))
    c = 0
    for i in range(1, len(pred) + 1):
        rel = 0
        if pred[i - 1] in target:
            rel = 1
            c += 1
        p_at_k[i - 1] = rel * c / i
    if c == 0:
        return 0.0
    else:
        return np.sum(p_at_k) / c


from operator import itemgetter


def gini_index(p):
    """
    计算Gini系数
    :param p: 物品流行度得分字典，key为物品ID，value为得分
    """
    j = 1
    n = len(p)
    G = 0
    for item, weight in sorted(p.items(), key=itemgetter(1)):
        G += (2 * j - n - 1) * weight
        j += 1
    return G / float(n - 1)


def popularity_score(records):
    """
    计算热门度得分
    :param records: 用户对物品的行为记录，每行格式为[user_id, item_id, score]
    """
    # 计算物品流行度得分
    item_count = {}
    for record in records:
        item_id = record[1]
        item_count[item_id] = item_count.get(item_id, 0) + 1
    # 计算Gini系数和热门度得分
    scores = list(item_count.values())
    if len(scores) == 0:
        return 0
    else:
        gini = gini_index(item_count)
        max_gini = 1 - 1 / float(len(scores))
        return 1 - gini / max_gini


def giniIndex(target, pred, api_freq):
    # print("{0}==>{1}".format(target, pred))
    H_pred = 0.0
    H_target = 0.0
    # + len(api_freq)
    all_pop = sum(api_freq)
    # print(sum((api_freq)))
    # pi = 0.0
    for i in range(1, len(pred) + 1):
        pi = (api_freq[pred[i - 1]]) / all_pop
        H_pred += pi * math.log(1 / pi, 2)

    for i in range(1, len(target)):
        pi = (api_freq[target[i - 1]]) / all_pop
        H_target += pi * math.log(1 / pi, 2)

    if H_target == 0.0 or H_pred == 0.0:
        print("---")

    # print("{0}==>{1}".format(H_target, H_pred))
    # return H_pred - H_target
    return H_target / H_pred, H_pred, H_target


def diversity(target, pred, api_tag_embed):
    n = len(pred)
    if n == 1:
        return 0
    div = 0.0
    for i in pred:
        for j in pred:
            if i == j:
                continue
            # div += Cosine(api_tag_embed[i], api_tag_embed[j])
            # div += Cosine(api_tag_embed[i], api_tag_embed[j])
            # div += torch.cosine_similarity(api_tag_embed[i], api_tag_embed[j], dim=0)
            # div += stats.pearsonr(api_tag_embed[i], api_tag_embed[j])[0]
            # union = set(list(np.where(api_tag_embed[i] == 1)[0])).union(list(np.where(api_tag_embed[j] == 1)[0]))
            itersection = set(list(np.where(api_tag_embed[i] == 1)[0])).intersection(
                list(np.where(api_tag_embed[j] == 1)[0]))

            if len(itersection) > 0:
                div += 1.0
            # div += 1.0 * len(itersection) / len(union)
            if div < 0:
                print("---")
    # return 1 - (div / (0.5 * n * (n - 1)))

    return (div * 2 / (n * (n - 1)))


def Cosine(dataA, dataB):
    sumData = dataA * dataB.T  # 若列为向量则为 dataA.T * dataB
    denom = np.linalg.norm(dataA) * np.linalg.norm(dataB)
    # 归一化
    return 0.5 + 0.5 * (sumData / denom)


def Pearson(dataA, dataB):
    # 皮尔逊相关系数的取值范围(-1 ~ 1),0.5 + 0.5 * result 归一化(0 ~ 1)
    return 0.5 + 0.5 * np.corrcoef(dataA, dataB, rowvar=0)[0][1]


def popular_bias(recommendations, popular_items):
    popular_recommendations = [rec for rec in recommendations if rec in popular_items]
    popular_bias = len(popular_recommendations) / len(recommendations)

    return popular_bias


def precision(target, pred):
    epsilon = 1e-8
    hit_set = list(set(target) & set(pred))
    return len(hit_set) / float(len(pred) + epsilon)


def recall(target, pred):
    epsilon = 1e-8
    hit_set = list(set(target) & set(pred))
    
    return len(hit_set) / float(len(target) + epsilon)
