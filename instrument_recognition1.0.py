# -*- coding = UTF-8 -*-
# @Author:何欣泽
# @Time:2020/7/11 22:23
# @File:yueqishibie.py
# @Software:PyCharm


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import operator


# 数据预处理
def pre_deal(filename):
    data_txt = open(filename, 'r', encoding='utf-8')

    data_list = data_txt.readlines()

    data_list[0] = data_list[0].lstrip('\ufeff')

    number_data = len(data_list)

    pre_mat = np.zeros((number_data, 3))

    label = []

    index = 0

    for line in data_list:
        line = line.strip()

        line = line.split('\t')

        pre_mat[index, :] = line[0:3]

        label.append(line[-1])

        index += 1

    return pre_mat, label


# 数据归一化
def Normalization(pre_mat):
    max_element = pre_mat.max(0)

    min_element = pre_mat.min(0)

    ranges = max_element - min_element

    length = pre_mat.shape[0]

    nor_mat = pre_mat - np.tile(min_element, [length, 1])

    nor_mat = nor_mat / np.tile(ranges, [length, 1])

    return nor_mat, ranges, min_element


# 数据可视化
def showdata(pre_mat, label):
    # 设置字体
    font = FontProperties(fname=r"c:\windows\fonts\STFANGSO.TTF", size=14)
    # 设置画布
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    number = 0
    flag = 0
    for i in label:
        if i == 'violin' and flag == 1:
            violin_number = number
            flag += 1
        elif i == 'sax' and flag == 0:
            sax_number = number
            flag += 1
        number += 1

    axs[0][0].hist(pre_mat[0:sax_number,0],color = 'orange',bins = 100,label = 'piano',alpha = 0.75)
    axs[0][0].hist(pre_mat[sax_number + 1:violin_number,0],color = 'red',bins = 100,label = 'sax',alpha = 0.75)
    axs[0][0].hist(pre_mat[violin_number:, 0], color= 'black', label='violin', bins = 100,alpha=0.75)
    axs[0][0].legend()
    axs0_title_text = axs[0][0].set_title(u'High frequency amplitude distribution', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'High frequency average amplitude', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'Average amplitude', FontProperties=font)
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')


    axs[0][1].hist(pre_mat[0:sax_number,1],color = 'orange',bins = 100,label = 'piano',alpha = 0.75)
    axs[0][1].hist(pre_mat[sax_number + 1:violin_number,1],color = 'red',bins = 100,label = 'sax',alpha = 0.75)
    axs[0][1].hist(pre_mat[violin_number:, 1], color= 'black', label='violin', bins = 100,alpha=0.75)
    axs[0][1].legend()
    axs0_title_text = axs[0][1].set_title(u'Low frequency amplitude distribution', FontProperties=font)
    axs0_xlabel_text = axs[0][1].set_xlabel(u'Low frequency average amplitude', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'Average amplitude', FontProperties=font)
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')


    axs[1][0].hist(pre_mat[0:sax_number,0],color = 'orange',bins = 100,label = 'piano',alpha = 0.75)
    axs[1][0].hist(pre_mat[sax_number + 1:violin_number,0],color = 'red',bins = 100,label = 'sax',alpha = 0.75)
    axs[1][0].hist(pre_mat[violin_number:, 0], color= 'black', label='violin', bins = 100,alpha=0.75)
    axs[1][0].legend()
    axs0_title_text = axs[1][0].set_title(u'Total frequency amplitude distribution', FontProperties=font)
    axs0_xlabel_text = axs[1][0].set_xlabel(u'Total frequency average amplitude', FontProperties=font)
    axs0_ylabel_text = axs[1][0].set_ylabel(u'Average amplitude', FontProperties=font)
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')


    plt.show()


def classifier(text_mat, train_mat, lable, k):
    line_number = train_mat.shape[0]

    temp_mat = np.tile(text_mat, [line_number, 1]) - train_mat

    temp_mat = temp_mat ** 2

    temp_mat = temp_mat.sum(axis=1)

    distances = temp_mat ** 0.5

    sort_distances = distances.argsort()

    lable_fit = []

    score_unit = 0.3

    score = {'violin': 0, 'sax': 0, 'piano': 0}

    for i in range(k):

        choose_lable = lable[sort_distances[i]]

        lable_fit.append(choose_lable)

        if choose_lable == 'violin':
            score['violin'] += score_unit * (k - i)
        elif choose_lable == 'sax':
            score['sax'] += score_unit * (k - i)
        elif choose_lable == 'piano':
            score['piano'] += score_unit * (k - i)

        score_unit -= 0.015

    score_result = sorted(score.items(), key=operator.itemgetter(1), reverse=True)

    # print(score_result)

    return score_result[0][0]


def main():
    filename = 'NEWDATA.txt'

    pre_mat, label = pre_deal(filename)

    nor_mat, ranges, min_element = Normalization(pre_mat)

    showdata(pre_mat, label)

    text_proportion = 0.05

    data_number = nor_mat.shape[0]

    wrong_proportion = []

    text_number = int(text_proportion * data_number)

    for k_number in range(2, 30):
        error_number = 0
        for i in range(text_number):

            classifier_result = classifier(nor_mat[i, :], nor_mat[text_number:data_number, :]
                                           , label[text_number:data_number], k_number)

            # print('分类结果是%s,实际结果是%s'%(classifier_result,label[i]))

            if classifier_result != label[i]:

                error_number += 1

        wrong_proportion.append(error_number / text_number)

        print('错误率%.2f%%' % ((error_number / text_number) * 100))
    k_number = range(2, 30)

    # 设置字体
    font = FontProperties(fname=r"c:\windows\fonts\STFANGSO.TTF", size=14)
    # 设置画布
    fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(13, 8))

    axs.scatter(x=k_number, y=wrong_proportion, color='black', s=30, alpha=.5)
    axs_title_text = axs.set_title(u'K的选定与错误率的关系', FontProperties=font)
    axs_xlabel_text = axs.set_xlabel(u'K值大小', FontProperties=font)
    axs_ylabel_text = axs.set_ylabel(u'错误率', FontProperties=font)
    plt.setp(axs_title_text, size=14, weight='bold', color='red')
    plt.setp(axs_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs_ylabel_text, size=12, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    main()
