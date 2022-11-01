from sklearn.metrics import accuracy_score
from sklearn import model_selection as cv
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rank(path):
    # load data
    print('load')
    data_path = path
    data_name = path.split("/")
    data_name = data_name[-1]

    data1 = pd.read_csv(data_path, header=0)
    data1 = data1.sort_values(by=["score"], ascending=False, inplace=False)
    data_2 = data1.iloc[:, 2:]
    warp_path = "E:/metabo/Wrapper-Feature-Selection-Toolbox-Python-main/data/"
    save_path = warp_path + "_fs_"+data_name
    data_2.to_csv(save_path, header=True, index=False)
    x = [len(data1) * 0.1, len(data1) * 0.1, len(data1) * 0.2, len(data1) * 0.3, len(data1) * 0.4, len(data1) * 0.5,
         len(data1) * 0.6, len(data1) * 0.7,
         len(data1) * 0.8, len(data1) * 0.9, len(data1)]
    x = [int(i) for i in x]
    acc_list = []

    for i in x:
        data_1 = data1.iloc[:i, :]
        data = pd.DataFrame(data_1.values.T, index=data_1.columns, columns=data_1.index)
        X = data.iloc[3:, 1:]
        X = X.astype(float)
        X = np.array(X)
        y = data.iloc[3:, 0]
        try:
            y = y.astype(float)
        except ValueError:
            pass
        else:
            pass
        y = np.array(y)

        ss = cv.KFold(n_splits=10, shuffle=True)

        for i in range(100):

            correct = 0
            # clf = KNeighborsClassifier(n_neighbors=5)
            clf = svm.SVC(kernel='rbf')

            for train, test in ss.split(X):
                # train a classification model with the selected features on the training dataset
                clf.fit(X[train], y[train])

                # predict the class labels of test data
                y_predict = clf.predict(X[test])

                # obtain the classification accuracy on the test data
                acc = accuracy_score(y[test], y_predict)
                correct = correct + acc

            acc_list.append(float(correct / 10))

    list = ["5%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    ave = []
    sum = 0
    for index, item in enumerate(acc_list):
        sum = sum + item
        if (index + 1) % 100 == 0:
            aven = sum / 100
            ave.append(aven)
            sum = 0
    print(ave)
    # list = [val for val in list for i in range(100)]
    s = pd.Series(list, name="surplus")

    df = pd.DataFrame(ave)
    df.insert(0, "surplus", s)
    result_path = "fs_result/" + "range_" + data_name
    df.to_csv(result_path)
    df.plot.box(title="f_score FS ACC")
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()


if __name__ == '__main__':
    rank('data/data_process/test.csv')
