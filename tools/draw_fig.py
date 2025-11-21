# _*_ coding:utf-8 _*_
"""
@Time     : 2022/12/9 21:59
@Author   : Wangxuanye
@File     : draw_fig.py
@Project  : MTFM
@Software : PyCharm
@License  : (C)Copyright 2018-2028, Taogroup-NLPR-CASIA
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2022/12/9 21:59        1.0             None
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import rcParams

# rcParams['font.family']='sans-serif'
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']

# def draw_ndcg(df):
#     global index
#     labels = [str[str.rfind('-') + 1:] for str in list(df.columns[2:8])]
#     index = np.arange(len(labels))
#     x = np.arange(len(labels))
#     x *= 6
#     # fig, ax = plt.subplots(figsize=(12, 8))
#     fig, ax = plt.subplots()
#     width = 0.35
#     group_size = len(df.T.columns)
#     for i in range(group_size):
#         vals = df.T[i][2:8]
#         if i < group_size / 2:
#             ax.bar(x - width * (group_size / 2 - i), vals, width, label=df.T[i][0])
#         elif i == group_size / 2:
#             ax.bar(x, vals, width, label=df.T[i][0])
#         else:
#             ax.bar(x + width * (i - group_size / 2), vals, width, label=df.T[i][0])
#     ax.set_ylabel('NDCG')
#     ax.set_xlabel('Top-N')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#     fig.tight_layout()
#     # plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
#     plt.legend(loc='upper left', fontsize=7)
#     plt.show()


# MAP 9：15
# Pre 16：22
# Recall 23：29

name2range = {
    "NDCG": [2, 8],
    "MAP": [9, 15],
    "Precision": [16, 22],
    "Recall": [23, 29],
    "F1": [30, 36],
}


def draw_pic(df, name="NDCG"):
    start, end = name2range[name]
    labels = [str[str.rfind('-') + 1:] for str in list(df.columns[start:end])]
    # index = np.arange(len(labels))

    group_size = len(df.T.columns)
    x = np.arange(len(labels))
    x = x * (group_size + 1) / 2
    # fig, ax = plt.subplots(figsize=(12, 8))
    fig, ax = plt.subplots()
    width = 0.35
    for i in range(group_size):
        vals = df.T[i][start:end]
        if i < group_size / 2:
            ax.bar(x - width * (group_size / 2 - i), vals, width, label=df.T[i][0])
        elif i == group_size / 2:
            ax.bar(x, vals, width, label=df.T[i][0])
        else:
            ax.bar(x + width * (i - group_size / 2), vals, width, label=df.T[i][0])
    ax.set_ylabel(name)
    ax.set_xlabel('Top-N')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    # plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
    # plt.legend(loc='upper left', fontsize=7)
    plt.legend(loc='upper right', fontsize=7)
    plt.show()


def draw_overall():
    # global data
    # df = pd.read_excel("../data/mashup-cold-start-ours.xlsx")
    df = pd.read_excel("../data/mashup-api-valid-percentage.xlsx")
    data = df.values
    print(data)
    # draw_ndcg(df)
    # draw_recall(df)
    draw_pic(df)
    draw_pic(df, "MAP")
    draw_pic(df, "Precision")
    draw_pic(df, "Recall")
    # draw_pic(df, "F1")


font_size = 12

figure(num=None, figsize=(2.8, 1.7), dpi=300)

def draw_all_RW():
    draw_RW("Programmable Web")
    draw_RW("Youshu")
    draw_RW("iFashion")
    pass

 
def draw_RW(name):
    x_RW = ["0", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]
    lables = {"NDCG@20": '-', "Recall@20": ':', "NDCG@40": '--', "Recall@40": '-.'}
    fn = "./data/mashup-bundle-RW-{0}.xlsx".format(name)
    df = pd.read_excel(fn)
    data = df.values
    print(data)
    fig, ax = plt.subplots()
    max = 0
    for k, pm in enumerate(data):
        lable = lables[pm[0]]
        # plt.plot(x_RW, pm[1:], linestyle=lable)
        ax.plot(x_RW, pm[1:], label=pm[0], linestyle=lable)  # axes对象绘图
        max = np.max([max, np.max(pm[1:])])
    # plt.legend([line1, line2], ["line 2", "line 1"], loc='best', frameon=False)

    width = max / len(x_RW)
    step = 0
    i = 1
    while step == 0:
        step = round(width, i)
        i += 1

    max = max + 2 * step

    plt.xticks(fontproperties='Times New Roman', fontsize=font_size)
    plt.yticks(np.arange(0, max, step), fontproperties='Times New Roman', fontsize=font_size)

    fig_name = "Performance metrics of {0}".format(name)
    plt.ylabel(fig_name, fontdict={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel(r"Regularization Weights $\lambda$", fontdict={'family': 'Times New Roman', 'size': font_size})
    ax.legend()

    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # fig.savefig("../data/Impact_of_Regularization_Weights.jpg")

    plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 10})
    # plt.grid(True)
    # plt.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.175)

    save_fn = "./data/Impact_of_Regularization_Weights_{0}.pdf".format(name)
    plt.savefig(save_fn, transparent=True, dpi=300)
    plt.show()


def draw_all_FD():
    draw_FD("Programmable Web")
    draw_FD("Youshu")
    draw_FD("iFashion")
    pass

 
def draw_FD(name):
    x_FD = ["32", "64", "128", "256", "512"]
    lables = {"NDCG@20": '-', "Recall@20": ':', "NDCG@40": '--', "Recall@40": '-.'}
    fn = "./data/mashup-bundle-FD-{0}.xlsx".format(name)
    df = pd.read_excel(fn)
    data = df.values
    print(data)
    fig, ax = plt.subplots()
    max = 0
    for k, pm in enumerate(data):
        lable = lables[pm[0]]
        # plt.plot(x_RW, pm[1:], linestyle=lable)
        ax.plot(x_FD, pm[1:], label=pm[0], linestyle=lable)  # axes对象绘图
        max = np.max([max, np.max(pm[1:])])
    # plt.legend([line1, line2], ["line 2", "line 1"], loc='best', frameon=False)

    width = max / len(x_FD)
    step = 0
    i = 1
    while step == 0:
        step = round(width, i)
        i += 1

    max = max + 2 * step

    plt.xticks(fontproperties='Times New Roman', fontsize=font_size)
    plt.yticks(np.arange(0, max, step), fontproperties='Times New Roman', fontsize=font_size)

    fig_name = "Performance metrics of {0}".format(name)
    plt.ylabel(fig_name, fontdict={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel(r"Feature Dimension", fontdict={'family': 'Times New Roman', 'size': font_size})
    ax.legend()

    # ax.yaxis.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    # fig.savefig("../data/Impact_of_Regularization_Weights.jpg")

    plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 10})
    # plt.grid(True)
    # plt.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.175)

    save_fn = "./data/Impact_of_Feature_Dims_{0}.pdf".format(name)
    plt.savefig(save_fn, transparent=True, dpi=300)
    plt.show()

def draw_ES():
    x_RW = ["8", "16", "32", "64", "128", "256", "512"]
    lables = {"NDCG": '-', "MAP": '--', "Precision": '-.', "Recall": ':'}
    df = pd.read_excel("../data/mashup-ES-lr0.001.xlsx")
    data = df.values
    print(data)
    fig, ax = plt.subplots()
    for k, pm in enumerate(data):
        lable = lables[pm[0]]
        # plt.plot(x_RW, pm[1:], linestyle=lable)
        ax.plot(x_RW, pm[1:], label=pm[0], linestyle=lable)  # axes对象绘图
    # plt.legend([line1, line2], ["line 2", "line 1"], loc='best', frameon=False)

    # plt.ylabel("Performance metrics")
    # plt.xlabel("Embedding Size of Mashup/API")

    plt.xticks(fontproperties='Times New Roman', fontsize=font_size)
    plt.yticks(np.arange(0, 1.1, 0.2), fontproperties='Times New Roman', fontsize=font_size)

    plt.ylabel("Performance metrics", fontdict={'family': 'Times New Roman', 'size': font_size})
    plt.xlabel("Embedding Size of Mashup/API", fontdict={'family': 'Times New Roman', 'size': font_size})

    ax.legend()
    plt.legend(loc='lower right', prop={'family': 'Times New Roman', 'size': 10})
    # fig.savefig("../data/Impact_of_Embedding_Size.jpg")
    plt.savefig("../data/Impact_of_Embedding_Size.svg", transparent=True, dpi=300)
    plt.show()


if __name__ == '__main__':
    # df = pd.read_excel("../data/mashup-clean.xlsx")
    # df = pd.read_excel("../data/mashup-clean-enhance.xlsx")
    # df = pd.read_excel("../data/mashup-cold-start.xlsx")
    # draw_overall()

    # draw_RW()
    # draw_ES()

    draw_all_RW()
    draw_all_FD()
    pass
    # fig = plt.figure(figsize=(16, 8))  # 设置画布大小
    #
    # group_size = len(df.T.columns)
    #
    # width = 0.1 * group_size
    # for i in range(group_size):
    #     vals = df.T[i][2:8]
    #     if i < group_size / 2:
    #         rect1 = plt.bar(index - i * width / 7, vals, width=0.1)
    #     else:
    #         rect2 = plt.bar(index + (i - group_size / 2) * width / 7, vals, width=0.1)
    #
    # plt.xlabel("Top-N")
    # plt.ylabel("NDCG")
    # plt.title("Complete Vocabulary Generated")
    # plt.xticks(ticks=index, labels=labels)
    #
