import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection as cv
from FS.gwo_co11_h2 import jfs  # change this to switch algorithm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn import svm
from time import *
import time
from itertools import chain
import seaborn as sns


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2) - 1
        return listnum[i]
    else:
        i = int(lnum / 2) - 1
        return (listnum[i] + listnum[i + 1]) / 2


def feature_select(data_path, particle_num, iteration_num):
    trick = str(time.time())
    data_name = data_path.split("/")
    data_name = data_name[1].replace(".csv", "")
    particle = particle_num
    iteration = iteration_num
    # load data
    data1 = pd.read_csv(data_path, header=0)
    data2 = pd.read_csv(data_path)
    data2 = data2.iloc[1:, :]
    data2.insert(0, 'ID', range(0, len(data2)))
    # print(data2)
    data = pd.DataFrame(data1.values.T, index=data1.columns, columns=data1.index)
    X = data.iloc[1:, 1:]  # data
    X = X.astype('float32')
    X = np.array(X)
    # print(X)
    y = data.iloc[1:, 0]
    try:
        y = y.astype('float32')
    except ValueError:
        pass
    else:
        pass
    y = np.array(y)

    feat = X
    label = y

    a_adv = []
    b_adv = []
    d_adv = []

    for n in particle:
        num_feature_list = []
        acc_list = []
        c = []
        for j in iteration:

            for i in range(30):
                # split data into train & validation (70 -- 30)
                # xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.25, stratify=label,
                #                                                 shuffle=True)
                fold = {'x': feat, 'y': label}

                # parameter
                k = 7
                N = n
                T = j
                opts = {'k': k, 'fold': fold, 'N': N, 'T': T}

                # perform feature selection
                # begin_time2 = time()
                start = time.time()
                fmdl = jfs(feat, label, opts)
                # end_time2 = time()
                # run_time2 = end_time2 - begin_time2
                # print('Run time of this iteration:', run_time2)
                sf = fmdl['sf']
                c.append(fmdl['convergence'])
                sf = list(sf)
                print(sf)
                # df = data2[data2["ID"].isin(sf)]
                # df.to_csv("test123_PSO.csv", header=True, index=False)

                print("#########done" + str(i) + "#########")

                # model with selected features
                # num_train = np.size(xtrain, 0)
                # num_valid = np.size(xtest, 0)
                x = X[:, sf]
                # y = y.reshape(num_train)  # Solve bug
                ss = cv.KFold(n_splits=10, shuffle=True)
                # x_train = xtrain[:, sf]
                # y_train = ytrain.reshape(num_train)  # Solve bug
                # x_valid = xtest[:, sf]
                # y_valid = ytest.reshape(num_valid)  # Solve bug

                # mdl = KNeighborsClassifier(n_neighbors=5)
                # mdl = svm.SVC()
                correct = []
                # mdl.fit(x_train, y_train)
                for r in range(10):
                    c2 = 0
                    mdl = KNeighborsClassifier(n_neighbors=k)
                    for train, test in ss.split(x):
                        # train a classification model with the selected features on the training dataset
                        mdl.fit(x[train], y[train])

                        # predict the class labels of test data
                        y_predict = mdl.predict(x[test])

                        # obtain the classification accuracy on the test data
                        acc = balanced_accuracy_score(y[test], y_predict)
                        c2 = c2 + acc
                    correct.append(c2 / 10)
                end = time.time()
                # accuracy
                # y_pred = mdl.predict(x_valid)
                # acc = accuracy_score(y_pred, y_valid)
                acc_list.append(mediannum(correct))
                # acc_list.append(acc)
                print("Accuracy:", 100 * mediannum(correct))
                print("运行时间:%.2f" % (end - start))
                # print("Accuracy:", 100 * acc)

                # number of selected features
                num_feat = fmdl['nf']
                num_feature_list.append(num_feat)
                print("Feature Size:", num_feat)
                print("#########done" + str(i) + "#########")

                a_adv.append(fmdl['a_adv'])
                b_adv.append(fmdl['b_adv'])
                d_adv.append(fmdl['d_adv'])

        convergence = sum(c) / len(c)

        ACC_std = np.std(np.array(acc_list), ddof=1)
        NUM_FEAT_std = np.std(np.array(num_feature_list), ddof=1)
        print("MeanACC-STD:")
        print(ACC_std, "\t", sum(acc_list) / len(acc_list))
        print("=" * 20)
        print("Best-ACC & Size:")
        print(max(acc_list), num_feature_list[acc_list.index(max(acc_list))])
        print("=" * 20)
        print("MeanSize-STD:")
        print(NUM_FEAT_std, "\t", sum(num_feature_list) / len(num_feature_list))
        print(convergence)
