import numpy as np
from numpy.random import rand
from FS.functionHO2 import Fun
import math
import random

random.seed()
sed = random.random()


def init_position(lb, ub, N, dim):
    Xh = np.zeros([N, int(dim * 0.05)], dtype='float')
    Xm = np.zeros([N, int(dim * 0.35)], dtype='float')
    Xl = np.zeros([N, (dim - (int(dim * 0.05) + int(dim * 0.35)))], dtype='float')
    for i in range(N):
        random.seed()
        l = int(random.uniform(0, int(dim * 0.2)))
        mid_number = int(l * random.uniform(0.8, 1))
        low_number = l - mid_number

        Xh[i, :] = random.uniform(0.5, 1)


        sample_list_m_1 = [i for i in range(int(dim * 0.35))]
        if len(sample_list_m_1) < mid_number:
            random.seed()
            Xm[i, :] = random.uniform(0.5, 1)

        else:
            sample_list_m = random.sample(sample_list_m_1, mid_number)

            random.seed()
            Xm[i, sample_list_m] = random.uniform(0.5, 1)


        sample_list_l = [i for i in range((dim - (int(dim * 0.05) + int(dim * 0.35))))]
        if len(sample_list_l) < low_number:
            random.seed()
            Xl[i, :] = random.uniform(0.5, 1)

        else:
            sample_list_l = random.sample(sample_list_l, low_number)

            random.seed()
            Xl[i, sample_list_l] = random.uniform(0.5, 1)

    X = np.hstack((Xh, Xl, Xm))

    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
            if result > sed:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
        if np.all(Xbin[i, :] == 0):
            for d in range(dim):
                result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
                if result > sed:
                    Xbin[i, d] = 1
                else:
                    Xbin[i, d] = 0
        if np.all(Xbin[i, :] == 1):
            for d in range(dim):
                result = 1 / (1 + math.exp(-10 * (X[i, d] - 0.5)))
                if result > sed:
                    Xbin[i, d] = 1
                else:
                    Xbin[i, d] = 0
    return Xbin


def binary_conversion2(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        Xbin[i, :] = X[i, :].copy()
        Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
        if np.all(Xbin[i, :] == 0):
            Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
        if np.all(Xbin[i, :] == 1):
            Xbin[i, :] = 1 * (1 / (1 + np.exp(-10 * (X[i, :] - 0.5))) >= sed)
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def jfs(xtrain, ytrain, opts):
    ub = 1
    lb = 0
    thres = 0.5
    mark = 0
    w = [1, 1]
    no_promoted = 0
    a_list = []
    alpha = 2
    F = 0.5  # factor
    max_index = 0

    N = opts['N']
    max_iter = opts['T']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration
    fit = np.zeros([N, 1], dtype='float')
    fit1 = np.zeros([N, 1], dtype='float')
    fit2 = np.zeros([N, 1], dtype='float')
    fit3 = np.zeros([N, 1], dtype='float')
    Xalpha = np.zeros([1, dim], dtype='float')
    Xbeta = np.zeros([1, dim], dtype='float')
    Xdelta = np.zeros([1, dim], dtype='float')
    Falpha = float('inf')
    Fbeta = float('inf')
    Fdelta = float('inf')
    num_list = []
    a_num = []
    a_fit = []
    fetnum_list = []
    fit_list = []
    error_list = []
    a_adv_list = []
    b_adv_list= []
    d_adv_list = []
    for i in range(N):
        fit[i, 0], num, error = Fun(xtrain, ytrain, Xbin[i, :], opts)
        num_list.append(int(num))
        fit_list.append(fit[i, 0])
        error_list.append(error)
        if fit[i, 0] < Falpha:
            Xalpha[0, :] = X[i, :]
            Falpha = fit[i, 0]
            mark = i

        if Fbeta > fit[i, 0] > Falpha:
            Xbeta[0, :] = X[i, :]
            Fbeta = fit[i, 0]

        if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
            Xdelta[0, :] = X[i, :]
            Fdelta = fit[i, 0]


    a_num.append(sum(num_list) / len(num_list))
    a_fit.append(sum(fit_list) / len(fit_list))
    fetnum_list.append(num_list[mark])

    # Pre
    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    curve[0, t] = Falpha.copy()

    t += 1

    while t < max_iter:
        flag = 0
        num_list = []
        fit_list = []
        a_max = 2
        a_min = 0
        a1 = 2 * (math.cos(0.5 * math.pi * ((t / max_iter) * (t / max_iter))))
        a2 = abs(math.cos(0.5 * math.pi * (t / max_iter)))

        a3 = 2 - abs(0.5 * math.cos(0.5 * math.pi * (t / max_iter)))

        for i in range(N):
            r1 = np.random.uniform(size=dim)
            r2 = np.random.uniform(size=dim)
            A1 = 2 * a1 * r1 - a1
            C1 = 2 * r2
            D1 = np.abs(C1 * Xalpha - X[i, :])
            X1 = (Xalpha - A1 * D1)

            r3 = np.random.uniform(size=dim)
            r4 = np.random.uniform(size=dim)
            # A2 = 2 * a2 * r3 - a2
            A2 = 2 * a2 * r3 - a2
            C2 = 2 * r4
            D2 = np.abs(C2 * Xbeta - X[i, :])
            X2 = (Xbeta - A2 * D2)

            r5 = np.random.uniform(size=dim)
            r6 = np.random.uniform(size=dim)

            A3 = 2 * a3 * r5 - a3
            C3 = 2 * r6
            D3 = np.abs(C3 * Xdelta - X[i, :])
            X3 = (Xdelta - A3 * D3)
            del r1, r2, r3, r4, r5, r6

            X[i, :] = (X1 + w[0] * X2 + w[1] * X3) / 3

            B = X[i, :].copy()
            B[B > 1] = 1
            B[B < 0] = 0
            X[i, :] = B

        Xbin = binary_conversion2(X, thres, N, dim)
        alpha_advance = 0
        beta_advance = 0
        delta_advance = 0

        # Fitness
        for i in range(N):
            fit[i, 0], num, error = Fun(xtrain, ytrain, Xbin[i, :], opts)
            num_list.append(num)
            fit_list.append(fit[i, 0])
            list_a = fit.tolist()
            max_index = list_a.index(max(list_a))
            if fit[i, 0] < Falpha:
                flag = 1
                alpha_advance = Falpha - fit[i, 0]
                Xalpha[0, :] = X[i, :]
                Falpha = fit[i, 0]
                mark = i

            if Fbeta > fit[i, 0] > Falpha:
                beta_advance = Fbeta - fit[i, 0]
                Xbeta[0, :] = X[i, :]
                Fbeta = fit[i, 0]

            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                delta_advance = Fdelta - fit[i, 0]
                Xdelta[0, :] = X[i, :]
                Fdelta = fit[i, 0]

        if beta_advance or delta_advance != 0:
            w[0] = beta_advance / (beta_advance + delta_advance)
            w[1] = delta_advance / (beta_advance + delta_advance)
        else:
            w[0] = 1
            w[1] = 1
        if alpha_advance == 0:
            no_promoted = no_promoted + 1
        else:
            no_promoted = 0


        B = np.zeros([3, dim], dtype='float32')
        BN = np.zeros([3, dim], dtype='float32')
        B[0, :] = Xalpha[0, :]
        B[1, :] = Xbeta[0, :]
        B[2, :] = Xdelta[0, :]
        BN[0, :] = 1 - Xalpha[0, :]
        BN[1, :] = 1 - Xbeta[0, :]
        BN[2, :] = 1 - Xdelta[0, :]


        for i in range(3):
            RN = np.random.permutation(3)
            for j in range(3):
                if RN[j] == i:
                    RN = np.delete(RN, j)
                    break
            r1 = RN[0]
            r2 = RN[1]

            B[i, :] = Xalpha[0, :] + F * (B[r1, :] - B[r2, :]) + F * (B[i, :] - X[max_index, :])


        B[B > 1] = 1
        B[B < 0] = 0
        Bbin = binary_conversion2(B, thres, 3, dim)
        if flag:
            fetnum_list.append(num_list[mark])
        else:
            fetnum_list.append(fetnum_list[t - 1])

        for i in range(3):
            fit[i, 0], num, error = Fun(xtrain, ytrain, Bbin[i, :], opts)
            if fit[i, 0] < Falpha:
                a_adv_list.append(t)
                Xalpha[0, :] = B[i, :]
                Falpha = fit[i, 0]
                mark = i
                fetnum_list[t] = num


            if Fbeta > fit[i, 0] > Falpha:
                beta_advance = Fbeta - fit[i, 0]
                # b_adv_list.append([t,beta_advance/Fbeta])
                b_adv_list.append(t)
                Xbeta[0, :] = B[i, :]
                Fbeta = fit[i, 0]

            if Fdelta > fit[i, 0] > Fbeta and fit[i, 0] > Falpha:
                delta_advance = Fdelta - fit[i, 0]
                # d_adv_list.append([t,delta_advance/Fdelta])
                d_adv_list.append(t)
                Xdelta[0, :] = B[i, :]
                Fdelta = fit[i, 0]

        if beta_advance or delta_advance != 0:
            w[0] = beta_advance / (beta_advance + delta_advance)
            w[1] = delta_advance / (beta_advance + delta_advance)
        else:
            w[0] = 1
            w[1] = 1


        curve[0, t] = Falpha.copy()
        a_num.append(sum(num_list) / len(num_list))
        a_fit.append(sum(fit_list) / len(fit_list))
        print("Iteration:", t + 1)
        print("Best (GWO):", curve[0, t])
        t = t + 1

    # Best feature subset
    Gbin = binary_conversion2(Xalpha, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    print(fetnum_list)


    print(a_adv_list,b_adv_list,d_adv_list)

    # Create dictionary
    gwo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'a_list': a_list,
                'convergence': mark / max_iter, 'num': a_num, 'fit': a_fit,
                'a_adv':a_adv_list, 'b_adv':b_adv_list,'d_adv': d_adv_list,
                'fet': fetnum_list}

    return gwo_data
