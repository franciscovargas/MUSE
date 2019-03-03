import numpy as np
import scipy.linalg as la
from sklearn.datasets import make_spd_matrix
# Routines for multiview linear FA


def nll_FA(S, W, Psi, n):
    """
    nll for mv-FA
    :param S[ndrray]: Cross covariance matrix (multiview covarince)
    :param W[ndrray]: W_i stack (W_i transformation from latent to obs)
    :param Psi[ndrray]: Block emission noise matrix
    :param n[int]: Number of datapoints
    :returns [float]: nll for mv-FA
    """

    num_view_features, latent_dim = W.shape

    C = np.dot(W, W.T) + Psi
    Cinv = la.inv(C) # Should be invertable

    s, det = np.linalg.slogdet(C)

    if s <= 0:
        print("Negative determinant of positive definite matrix")
        import ipdb
        ipdb.set_trace()
    trace_term = np.trace(np.dot(Cinv, S))

    return 0.5 * n * (latent_dim * np.log(2 * np.pi) + det + trace_term)


def chunk(A, i, j, nviews):
    D = A.shape[0] // nviews
    return A[i * D:(i+1) * D, j * D:(j+1) * D]


def init_with_browne(view_list, latent_dim, nviews, S):
    T = la.block_diag(*[np.linalg.cholesky(np.cov(x.T)) for x in view_list])

    Tinv = la.inv(T)
    V = Tinv.dot(S).dot(Tinv.T)

    # V should be symmetric
    assert np.all(np.isclose(V, V.T, rtol=1e-10))
    D, U = la.eigh(V)

    # Pick latent_dim largest (eigh returns small to large)
    U = U[:, -latent_dim:]
    D = D[-latent_dim:]

    # Turn D into diagonal matrix
    D = np.diag(D)

    I = np.eye(latent_dim)
    gamma_0 = np.sqrt(nviews / (nviews - 1.0)) * U.dot(np.sqrt(D - I))

    gammas = [chunk(gamma_0, i, 0, nviews) for i in range(nviews)]

    Ws = [chunk(T, i, i, nviews).dot(gammas[i]) for i in range(nviews)]
    Psis = [chunk(S, i, i, nviews) - Ws[i].dot(Ws[i].T) for i in range(nviews)]

    return Ws, Psis


def em_multi_fa(view_list, iterations=100, latent_dim=None, debug=False, initialise_at_guess=False):
    """
    Em Algorithm for multi-view model (Section 4.1  of https://www.di.ens.fr/~fbach/probacca.pdf):
                            z ~ N(0,1)
                            x_i = W_i * z + e_i
                            e_i ~ N(mu_i, Psi_i)
    :param view_list[list[ndarray]]: contains design matrices from multiple
                                     views
    :param iterations[int]: integer number of iterations
    :param latent_dim[int]: dimensionality of latent space z
    :returns [list[ndarray]]: Linear transformations
                              from latent space to view
    """

    # Check viewlist is not empty
    if not view_list: raise BaseException("View list cant be empty")

    nviews = len(view_list)
    n = view_list[0].shape[0] # Number of datapoints

    # If no latent dim provided default to dim of smalest view
    if latent_dim is None: latent_dim = min([ x.shape[1] for x in view_list ])

    concat_dims = np.concatenate(view_list, axis=1) # Concatenating dimensions
    Sigma = np.cov(concat_dims.T)  # Cross correlation covariance matrix

    if not initialise_at_guess:
        # Matrix with linear transformations from latent space
        W = np.random.randn( concat_dims.shape[1], latent_dim )

        # Block diag matrix for emission noise covariance
        Psi = la.block_diag(*[ make_spd_matrix(x.shape[1]) for x in view_list])
    else:
        Ws, Psis = init_with_browne(view_list, latent_dim, nviews, Sigma)
        W = np.concatenate(Ws, axis=0)
        Psi = la.block_diag(*Psis)

    loss = nll_FA(Sigma, W, Psi, n)
    print("Starting loss", loss)

    # Making block diag (will keep using psi_mask)
    Psi_mask = ~(la.block_diag(*[ np.ones((x.shape[1], x.shape[1])) for x in view_list ]) == 1.0 )

    Iz = np.eye(latent_dim)
    reg = Iz * 1e-10
    losses = []
    # EM
    from tqdm import tqdm
    for i in tqdm(range(iterations)):

        # Setup var/matrices
        Psi_inv = la.inv(Psi)

        M_t = la.inv( Iz + W.T.dot(Psi_inv).dot(W) + reg)

        # Enforce symmetry in M_t
        M_t = (M_t + M_t.T) / 2

        # Calculating W_{t+1}
        num = Sigma.dot(Psi_inv).dot(W).dot(M_t)

        reciprocal = la.inv(M_t + M_t.dot(W.T).dot(Psi_inv).dot(Sigma).dot(Psi_inv).dot(W).dot(M_t) + reg)

        # Enforce symmetry in M_t
        reciprocal = (reciprocal + reciprocal.T) / 2

        W_new = num.dot(reciprocal) # W_{t+1}

        # Calculating Psi_t
        Psi_new = Sigma - num.dot(W_new.T)

        # Enforce symmetry in Psi
        Psi_new = (Psi_new + Psi_new.T) / 2

        Psi_new[Psi_mask] = 0.0  # Decorrelating emmission noise (block diag)

        loss = nll_FA(Sigma, W_new, Psi_new, n)

        if i % (iterations // 10) == 0:
            print(loss)

        # print(loss)

        W = W_new

        Psi = Psi_new

        if debug:
            losses.append(loss)

    print(loss)

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.show()

    W_opt_list = [W[i*view_list[i].shape[1]:(i + 1) * view_list[i].shape[1], :] for i in range(nviews)]
    Psi_opt_list = [chunk(Sigma, i, i, nviews) - np.dot(W_opt_list[i], W_opt_list[i].T) for i in range(nviews)]
    mus = [np.mean(view_list[i], axis=0) for i in range(nviews)]

    return W_opt_list, Psi_opt_list, mus


def pcca(Sigma, latent_dim, n, return_components=False):
    Sigma_11 = chunk(Sigma, 0, 0, 2)
    Sigma_22 = chunk(Sigma, 1, 1, 2)
    Sigma_12 = chunk(Sigma, 0, 1, 2)

    Sigma_11_inv_sqrt = la.sqrtm(la.inv(Sigma_11))
    Sigma_22_inv_sqrt = la.sqrtm(la.inv(Sigma_22))

    Sigma_prod = Sigma_11_inv_sqrt.dot(Sigma_12).dot(Sigma_22_inv_sqrt)

    V1, L, V2 = la.svd(Sigma_prod)

    # SVD returns this transposed
    V2 = V2.T

    U1, U2 = Sigma_11_inv_sqrt.dot(V1), Sigma_22_inv_sqrt.dot(V2)

    # Pick latent_dim largest (svd returns large to small)
    L = L[:latent_dim]
    U1 = U1[:, :latent_dim]
    U2 = U2[:, :latent_dim]

    assert L[0] >= L[1]

    P_sqrt = np.diag(np.sqrt(L))

    # # Assuming identity rotation
    W1_opt = Sigma_11.dot(U1).dot(P_sqrt)
    W2_opt = Sigma_22.dot(U2).dot(P_sqrt)

    Psi1_opt = Sigma_11 - W1_opt.dot(W1_opt.T)
    Psi2_opt = Sigma_22 - W2_opt.dot(W2_opt.T)

    W_opt = np.concatenate([W1_opt, W2_opt], axis=0)
    Psi_opt = la.block_diag(*[Psi1_opt, Psi2_opt])
    loss_opt = nll_FA(Sigma, W_opt, Psi_opt, n)
    print("opt loss", loss_opt)

    if return_components:
        return [W1_opt, W2_opt], [Psi1_opt, Psi2_opt], U1, U2, P_sqrt

    return [W1_opt, W2_opt], [Psi1_opt, Psi2_opt]


def guess_for_closed_form_fa(view_list, latent_dim=None, debug=False, return_components=False):
    """
    Guess for closed form for multi-view model (Section 4.1  of https://www.di.ens.fr/~fbach/probacca.pdf):
                            z ~ N(0,1)
                            x_i = W_i * z + e_i
                            e_i ~ N(mu_i, Psi_i)
    :param view_list[list[ndarray]]: contains design matrices from multiple
                                     views
    :param iterations[int]: integer number of iterations
    :param latent_dim[int]: dimensionality of latent space z
    :returns [list[ndarray]]: Linear transformations
                              from latent space to view
    """
    # Check viewlist is not empty
    if not view_list: raise BaseException("View list cant be empty")

    nviews = len(view_list)
    n = view_list[0].shape[0]  # Number of datapoints

    mus = [np.mean(view_list[i], axis=0) for i in range(nviews)]

    # If no latent dim provided default to dim of smalest view
    if latent_dim is None: latent_dim = min([x.shape[1] for x in view_list])

    concat_dims = np.concatenate(view_list, axis=1)  # Concatenating dimensions
    Sigma = np.cov(concat_dims.T)  # Cross correlation covariance matrix

    if nviews == 2:
        result = pcca(Sigma, latent_dim, n, return_components)

        if return_components:
            W_opt_list, Psi_opt_list, U1, U2, P_sqrt = result
            return W_opt_list, Psi_opt_list, mus, U1, U2, P_sqrt
        W_opt_list, Psi_opt_list = result
        return W_opt_list, Psi_opt_list, mus

    raise NotImplementedError("To be implemented")


def to_latent(X, W_x, Psi_x, mu_x):
    """
    :param X[ndarray]: Design matrix for a single view
    :param Wx[ndarray]: View generation matrix from latent (for a single view)
    :return Z[ndarray]:  E[Z | X, Wx] Latent embedding design matrix
    """

    Ih = np.eye(W_x.shape[1])

    X_transformed = X - mu_x

    Wpsi = W_x.T.dot(np.linalg.inv(Psi_x))

    cov_z = la.inv(Ih + np.dot(Wpsi, W_x))
    tmp = cov_z.dot(Wpsi)

    X_transformed = np.dot(X_transformed, tmp.T)

    return X_transformed
