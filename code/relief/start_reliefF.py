
import pandas as pd
import numpy as np
from skfeature.function.similarity_based import reliefF
from score_range import rank
import matplotlib.pyplot as plt


def relief_begin(path):
    data_path = path.split("/")
    data_name = data_path[-1]
    f_name = data_path[0]
    # load data
    data1 = pd.read_csv(path, header=0)
    data = pd.DataFrame(data1.values.T, index=data1.columns, columns=data1.index)
    print(data)
    X = data.iloc[1:, 1:]  # data
    X = X.astype(float)
    X = np.array(X)
    y = data.iloc[1:, 0]
    y = np.array(y)

    score = reliefF.reliefF(X, y, mode="raw")
    score = np.append(float("inf"), score)
    s = pd.Series(score, name="score")
    data1.insert(0, "score", s)
    save_path = f_name + "/data_process/" + "fs_reliefF"+data_name
    data1.to_csv(save_path)
    return save_path


if __name__ == '__main__':
    path = relief_begin("data/DLBCL.csv")
    rank(path)

