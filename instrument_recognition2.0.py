#-*- coding = utf-8 -*-
#@Author:何欣泽
#@Time:2020/7/14 22:16
#@File:new_data_shibie.py
#@Software:PyCharm



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

    pre_mat = np.zeros((number_data, 18))

    label = []

    index = 0

    for line in data_list:

        line = line.strip()

        line = line.split('\t')

        pre_mat[index, :] = line[0:18]

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

    LabelsColors = []
    for i in label:
        if i == 'violin':
            LabelsColors.append('orange')
        if i == 'piano':
            LabelsColors.append('red')
        if i == 'sax':
            LabelsColors.append('black')

    # 画出散点图,以第一列低频区、第二列高频区数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=pre_mat[:, 0], y=pre_mat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置格式
    axs0_title_text = axs[0][0].set_title(u'低频中平均振幅和标准差关系', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'平均振幅大小', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'标准差', FontProperties=font)
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')

    # 画出散点图,以第一列低频区、第二列总频区数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=pre_mat[:, 6], y=pre_mat[:, 9], color=LabelsColors, s=15, alpha=.5)
    # 设置格式
    axs0_title_text = axs[0][1].set_title(u'高频平均振幅和最大值大小关系', FontProperties=font)
    axs0_xlabel_text = axs[0][1].set_xlabel(u'高频区振幅大小', FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'高频区最大值大小', FontProperties=font)
    plt.setp(axs0_title_text, size=14, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')

    # 画出散点图,以第一列高频区、第二列总频区数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=pre_mat[:, 1], y=pre_mat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置格式
    axs0_title_text = axs[1][0].set_title(u'总频和高频振幅大小关系', FontProperties=font)
    axs0_xlabel_text = axs[1][0].set_xlabel(u'高频区振幅大小', FontProperties=font)
    axs0_ylabel_text = axs[1][0].set_ylabel(u'总振幅大小', FontProperties=font)
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

        score_unit -= 0.01

    score_result = sorted(score.items(), key=operator.itemgetter(1), reverse=True)

    # print(score_result)

    return score_result[0][0]


def main():
    filename = 'all_data.txt'

    pre_mat, label = pre_deal(filename)

    nor_mat, ranges, min_element = Normalization(pre_mat)

    showdata(pre_mat, label)

    text_proportion = 0.1

    data_number = nor_mat.shape[0]

    wrong_proportion = []

    text_number = int(text_proportion * data_number)

    for k_number in range(2, 30):
        error_number = 0.0
        for i in range(text_number):

            classifier_result = classifier(nor_mat[i, :], nor_mat[text_number:data_number, :]
                                           , label[text_number:data_number], k_number)

            # print('分类结果是%s,实际结果是%s'%(classifier_result,label[i]))

            if classifier_result != label[i]:

                error_number += 1

            wrong_proportion_temp = float(error_number / text_number)
        wrong_proportion.append(wrong_proportion_temp)

        print('当K=%d时,测试样本个数为%d,出错个数为%d,错误率%.2f%%' % (k_number,text_number,error_number,(error_number / text_number) * 100))
    k_number = range(2, 30)

    # 设置字体
    font = FontProperties(fname=r"c:\windows\fonts\STFANGSO.TTF", size=14)
    # 设置画布
    fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(13, 8))

    axs.scatter(x=k_number, y=wrong_proportion, color='red', s=50,marker='+')
    axs_title_text = axs.set_title(u'The relationship between K value and Error rate', FontProperties=font)
    axs_xlabel_text = axs.set_xlabel(u'K value', FontProperties=font)
    axs_ylabel_text = axs.set_ylabel(u'Error Rate', FontProperties=font)
    plt.setp(axs_title_text, size=14, weight='bold', color='red')
    plt.setp(axs_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs_ylabel_text, size=12, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    main()
