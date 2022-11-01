import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection as cv
from sklearn.metrics import accuracy_score
from sklearn import svm


# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k = opts['k']
    fold = opts['fold']
    xtrain = fold['x']
    ytrain = fold['y']
    num = np.size(xtrain, 0)
    xtrain = xtrain[:, x == 1]
    ytrain = ytrain.reshape(num)

    ss = cv.KFold(n_splits=10, shuffle=True)
    clf = KNeighborsClassifier(n_neighbors=5)
    correct = 0

    for train, test in ss.split(xtrain):
        clf.fit(xtrain[train], ytrain[train])

        # predict the class labels of test data
        y_predict = clf.predict(xtrain[test])

        # obtain the classification accuracy on the test data
        acc = accuracy_score(ytrain[test], y_predict)
        correct = correct + acc

    error = 1 - float(correct / 10)

    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    global error
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0 or num_feat == 1:
        cost = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost, num_feat, error
