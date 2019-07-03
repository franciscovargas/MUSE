import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
# from data.johnson import data as johnson_data


def main():
    np.random.seed(100)

    c, s = lambda theta: np.cos(theta), lambda theta: np.sin(theta)
    R = lambda theta: np.array(
        (
            (c(theta), -s(theta)),
            (s(theta), c(theta))
        )
    )

    # print(johnson_data)

    R_x = R(-30)
    R_y = R(30)
    R_w = R(-10)

    ### ND DATA
    D = 2
    VAL = 1000
    TRN = 1000000
    T = TRN
    N = VAL + TRN


    Z = np.random.randn(N, D)

    c = 0.03
    cs = np.abs(np.random.randn(3) * 0.003 + c)
    print(cs)

    e_x = cs[0] * np.random.randn(N, D)
    e_y = cs[1] * np.random.randn(N, D)
    e_w = cs[2] * np.random.randn(N, D)

    A = np.random.randn(D, D)
    B = np.random.randn(D, D)
    C = np.random.randn(D, D)
    # A = R_x
    # B = R_y
    # C = R_w

    #
    # # TODO: THIS MAKES IT UNLEARNABLE, WHY?
    # # A, B, C = map(lambda X: X.dot(X.T) / 10, [A, B, C])
    #
    X = Z.dot(A) + 0*e_x + 0.0000000010 * np.random.randn(1, D)
    Y = Z.dot(B) + 0*e_y + 0.0000000010 * np.random.randn(1, D)
    W = Z.dot(C) + 0*e_w + 0.0000000010 * np.random.randn(1, D)

    dico_keys = list(range(N))
    dico_keys_train = dico_keys[:T]
    dico_keys_val = dico_keys[T:]

    Z_train = Z[:T]
    X_train = X[:T]
    Y_train = Y[:T]
    W_train = W[:T]

    pd.DataFrame(X, index=dico_keys).to_csv("data/x_vecs.csv", header=[N,D] + [None] * (D-2), sep=" ")
    pd.DataFrame(Y, index=dico_keys).to_csv("data/z_vecs.csv", header=[N,D] + [None] * (D-2), sep=" ")
    pd.DataFrame(Y, index=dico_keys).to_csv("data/y_vecs.csv", header=[N,D] + [None] * (D-2), sep=" ")
    pd.DataFrame(W, index=dico_keys).to_csv("data/w_vecs.csv", header=[N,D] + [None] * (D-2), sep=" ")

    pd.DataFrame(dico_keys, index=dico_keys).to_csv("data/crosslingual/dictionaries/f1-f2.txt", header=False, sep=" ")
    pd.DataFrame(dico_keys_train, index=dico_keys_train).to_csv("data/crosslingual/dictionaries/f1-f2.0-5000.txt", header=False, sep=" ")
    pd.DataFrame(dico_keys_val, index=dico_keys_val).to_csv("data/crosslingual/dictionaries/f1-f2.5000-6500.txt", header=False, sep=" ")

    ploter = True

    if ploter:

        if X.shape[-1] == 2:
            ### 2D PLOTTING
            plt.plot(*X.T, linestyle='None', marker='.')
            plt.plot(*Y.T, linestyle='None', marker='.')
            plt.plot(*W.T, linestyle='None', marker='.')

        plt.show()



if __name__ == "__main__" :
    main()
