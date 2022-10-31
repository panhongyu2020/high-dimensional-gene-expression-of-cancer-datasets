import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection as cv
from FS.gwo_co11_h2 import jfs  # change this to switch algorithm
from sklearn.metrics import balanced_accuracy_score
from time import *
import time



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

    data1 = pd.read_csv(data_path, header=0)
    data2 = pd.read_csv(data_path)
    data2 = data2.iloc[1:, :]
    data2.insert(0, 'ID', range(0, len(data2)))

    data = pd.DataFrame(data1.values.T, index=data1.columns, columns=data1.index)
    X = data.iloc[1:, 1:]  # data
    X = X.astype('float32')
    X = np.array(X)

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

            for i in range(20):

                fold = {'x': feat, 'y': label}

                # parameter
                k = 5  # k-value in KNN
                N = n  # number of particles
                T = j  # maximum number of iterations
                opts = {'k': k, 'fold': fold, 'N': N, 'T': T}


                start = time.time()
                fmdl = jfs(feat, label, opts)

                sf = fmdl['sf']
                c.append(fmdl['convergence'])
                sf = list(sf)
                print(sf)


                print("#########done" + str(i) + "#########")


                x = X[:, sf]

                ss = cv.KFold(n_splits=10, shuffle=True)

                correct = []

                for r in range(10):
                    c2 = 0
                    mdl = KNeighborsClassifier(n_neighbors=5)
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

                acc_list.append(mediannum(correct))

                print("Accuracy:", 100 * mediannum(correct))
                print("time:%.2f" % (end - start))


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

