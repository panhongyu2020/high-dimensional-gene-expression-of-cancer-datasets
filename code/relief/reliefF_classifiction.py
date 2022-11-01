from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection as cv
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():

    # load data
    data1 = pd.read_csv('data/data_process/DLBCL.csv',header=0)
    data_y = pd.DataFrame(data1.values.T, index=data1.columns, columns=data1.index)
    acc_list = []

    # data = data1[(data1["rank"] <= len(data1)*0.4)&(data1["rank"]>=len(data1)*0.3)]
    data = data1[(data1["rank"] <= 500)]

    data_2 = data.iloc[:,2:]
    data_2.to_csv("data/reliefF-DLBCL.csv",header = True,index = False)

    data_x = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
    X = data_x.iloc[3:, 1:]
    X = X.astype(float)
    X = np.array(X)
    y = data_y.iloc[3:, 0]
    print(y)
    y = y.astype(float)
    y = np.array(y)

    ss = cv.KFold(n_splits=10, shuffle=True)


    c2 = 0

    for i in range(100):

        correct = 0

        clf = KNeighborsClassifier(n_neighbors=5)

        for train, test in ss.split(X):
            # train a classification model with the selected features on the training dataset
            clf.fit(X[train], y[train])

            # predict the class labels of test data
            y_predict = clf.predict(X[test])

            # obtain the classification accuracy on the test data
            acc = accuracy_score(y[test], y_predict)
            correct = correct + acc
        c2 = c2 + correct / 10

        acc_list.append(float(correct / 10))
    print(c2)

if __name__ == '__main__':
        main()

