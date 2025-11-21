# _*_ coding:utf-8 _*_
"""
@Time     : 2023/9/14 22:24
@Author   : Wangxuanye
@File     : case_study.py
@Project  : FSFM
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/9/14 22:24        1.0             None
"""

import json


def swap(list1, list2):
    return list2, list1


def read_json(json_name):
    intersection_result = {}
    recommed_list = {}
    target_list = {}

    # 读取JSON文件
    with open(json_name, 'r') as f:
        data = json.load(f)

    # 按照最外层的 [ ] 括号进行区分
    for item in data:
        # 获取第一个括号内的元素列表和第二个括号内的元素列表
        if json_name == "../model/case/RWR-8x.json":
            name, list1, list2, _ = item
        else:
            name, list1, list2 = item
        # 计算两个列表的交集
        if len(list1) == 13176:
            list1, list2 = swap(list1, list2)

        intersection = list(set(list1) & set(list2[:5]))
        # print(name, intersection)
        intersection_result[name] = intersection
        recommed_list[name] = list2[:5]
        target_list[name] = list1

    # print(intersection_result)
    return intersection_result, recommed_list, target_list


if __name__ == '__main__':
    # sehgcn, pre_sehgcn, target = read_json(
    #     "../model/case/SM-NoMF-2GCN-plus-tanh-lr0.001-x8.0-withcat-dim_128-l2_1e-07-rdlink-rdmsk.json")
    # fsfm, pre_fsfm, _ = read_json("../model/case/FSFM-NoMF-2GCN-diversity-lr0.0003-x8-withcat-dim_128-l2_1e-07.json")
    # mtfm, pre_mtfm, _ = read_json("../model/case/MTFM-lr0.001-x8-withcat-l2_1e-07.json")
    # rwr, pre_rwr, _ = read_json("../model/case/RWR.json")
    # spr, pre_spr, _ = read_json("../model/case/SPR-x8.json")
    # lstm, pre_lstm, _ = read_json("../model/case/LSTM-x8-lr0.0001.json")
    # pop, pre_pop, _ = read_json("../model/case/Pop-x8.json")
    # ncf, pre_ncf, _ = read_json("../model/case/NCF-coldStart-x8-rdmk.json")
    # cf, pre_cf, _ = read_json("../model/case/CF-coldStart-x8.json")

    sehgcn, pre_sehgcn, target = read_json(
        "../model/case/SM-plus-tanh-lr0.001-x1.0-withcat-dim_128-l2_1e-07-no-rdlink-rdmsk.json")
    fsfm, pre_fsfm, _ = read_json("../model/case/FSFM-NoMF-2GCN-diversity-lr0.0003-x8-withcat-dim_128-l2_1e-07.json")
    mtfm, pre_mtfm, _ = read_json("../model/case/MTFM-lr0.001-x1-withcat-l2_1e-07.json")
    rwr, pre_rwr, _ = read_json("../model/case/RWR-8x.json")
    spr, pre_spr, _ = read_json("../model/case/SPR-x1.json")
    lstm, pre_lstm, _ = read_json("../model/case/LSTM-x1-lr0.0001.json")
    pop, pre_pop, _ = read_json("../model/case/Pop-x1.json")
    ncf, pre_ncf, _ = read_json("../model/case/NCF-coldStart-x1.json")
    cf, pre_cf, _ = read_json("../model/case/CF-coldStart-x1.json")

    print("doing....")
    # # 打印name及其出现次数
    # and \
    #         len(mtfm[name]) >= len(cf[name]) and \
    #                 len(mtfm[name]) >= len(cf[name])
    # len(rwr[name]) and len(rwr[name]) >= len(spr[name]) and len(spr[name]) >= len(
    #     lstm[name]) and len(lstm[name]) >= len(ncf[name]) and len(lstm[name]) >=
    # and len(sehgcn[name]) >= 1 and len(mtfm[name]) > 1
    for name in sehgcn:
        if len(sehgcn[name]) > len(fsfm[name]) and len(sehgcn[name]) >= 1 and len(sehgcn[name]) > len(mtfm[name]) and len(sehgcn[name]) >= len(rwr[name]) and len(mtfm[name]) > 0:
            print("---")
            print(name, len(sehgcn[name]))
            print("{0}-->{1}".format(pre_sehgcn[name], len(sehgcn[name])))
            print("{0}-->{1}".format(pre_fsfm[name], len(fsfm[name])))
            print("{0}-->{1}".format(pre_mtfm[name], len(mtfm[name])))

            print("{0}-->{1}".format(pre_rwr[name], len(rwr[name])))
            print("{0}-->{1}".format(pre_spr[name], len(spr[name])))
            print("{0}-->{1}".format(pre_lstm[name], len(lstm[name])))
            print("{0}-->{1}".format(pre_pop[name], len(pop[name])))
            print("{0}-->{1}".format(pre_ncf[name], len(ncf[name])))
            print("{0}-->{1}".format(pre_cf[name], len(cf[name])))

            # print(pre_fsfm[name])
            # print(pre_mtfm[name])
            # print(pre_rwr[name])
            # print(pre_spr[name])
            # print(pre_lstm[name])
            # print(pre_pop[name])
            # print(pre_ncf[name])
            # print(pre_cf[name])
            # print(pre_cf[name])

            print("ground truth:")
            print(target[name])
