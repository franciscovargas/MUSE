import numpy as np
from src.fasttext import FastVector
from scipy.spatial.distance import cdist
import scipy.linalg as la


def read_dict(dict_file):
    """
    :param dict_file[OrderedDict]: train/test dictionary from Dinu

    :param test_dict[OrderedDict]: language pair dict < source, target >
    """
    return [tuple(line.strip().split()) for line in open(dict_file)]


def sample_language_pair(Ws, Psis, mus, X_dict, Y_dict, intersection_wordset):
    L_x = np.linalg.cholesky(Psis[0])
    L_y = np.linalg.cholesky(Psis[1])
    Xmat = X_dict.embed
    Ymat = Y_dict.embed

    Z = np.random.randn(2000, 300)
    e1 = np.random.randn(2000, 300)
    e2 = np.random.randn(2000, 300)

    e_x = e1.dot(L_x.T)
    e_y = e2.dot(L_y.T)
    X = Z.dot(Ws[0].T) + mus[0] + e_x
    Y = Z.dot(Ws[1].T) + mus[1] + e_y

    print("Started cdists")

    Xs = cdist(X, Xmat[:50000,:], metric='cosine')
    Xindices = np.argmin(Xs, axis=1)

    print("Done sampling Xs")

    Ys = cdist(Y, Ymat[:50000,:], metric='cosine')
    Yindices = np.argmin(Ys, axis=1)

    print("Done sampling Ys")

    import pprint

    pp = pprint.PrettyPrinter(depth=2)

    x_pair = np.array(X_dict.id2word)[Xindices]
    y_pair = np.array(Y_dict.id2word)[Yindices]

    x_embds = X_dict.embed[Xindices]
    y_embds = Y_dict.embed[Yindices]

    concat_embds = np.concatenate([x_embds, y_embds], axis=1)
    concat_mean = np.concatenate(mus, axis=0)

    W = np.concatenate(Ws, axis=0)
    C = W.dot(W.T) + la.block_diag(*Psis)
    C_inv = np.linalg.inv(C)

    L_inv = np.linalg.cholesky(C_inv)

    concat_embds -= concat_mean

    concat_embds = np.dot(concat_embds, L_inv.T)

    scores = np.linalg.norm(concat_embds, axis=1)

    indices = np.argsort(scores)

    out = list(zip(list(x_pair[indices]), list(y_pair[indices])))

    out_filtered = [tup for tup in out if tup not in intersection_wordset]

    print(len(out))
    print(len(out_filtered))

    pp.pprint(out_filtered)

    return out_filtered


def main():
    np.random.seed(100)

    src = 'en'
    tgt = 'es'

    pcca = True

    sourcerer = 'data/'

    en_dictionary = FastVector(vector_file=sourcerer + 'wiki.{0}.vec'.format(src))
    it_dictionary = FastVector(vector_file=sourcerer + 'wiki.{0}.vec'.format(tgt))

    print("Done allocating dicts")

    if tgt == 'it':
        intersection_wordset = read_dict("data/crosslingual/dictionaries/OPUS_en_it_europarl_train_5K.txt") + read_dict(
            "data/crosslingual/dictionaries/OPUS_en_it_europarl_test.txt")
    else:
        intersection_wordset = read_dict("data/crosslingual/dictionaries/{0}-{1}.txt".format(src, tgt))
    print(len(intersection_wordset))
    intersection_wordset = [(u, v) for u, v in intersection_wordset if u in en_dictionary and v in it_dictionary]
    print(len(intersection_wordset))

    print("Done reading word pairs")

    source = np.zeros((len(intersection_wordset), en_dictionary.n_dim))
    target = np.zeros((len(intersection_wordset), it_dictionary.n_dim))

    for i, word in enumerate(intersection_wordset):
        source[i] = en_dictionary[word[0]]
        target[i] = it_dictionary[word[1]]

    print("Started training")

    if pcca:
        from src.alignment_functions import guess_for_closed_form_fa, to_latent
        Ws, Psis, mus = guess_for_closed_form_fa([source, target])

        tups = sample_language_pair(Ws, Psis, mus, en_dictionary, it_dictionary, intersection_wordset)

        ff = open('samples_{0}_{1}.txt'.format(src, tgt), 'w')
        ff.write("\n".join([" ".join(tup) for tup in tups]))
        ff.close()

        print("Done training, starting transform")
        source_latents = to_latent(en_dictionary.embed, Ws[0], Psis[0], mus[0])
    else:
        from src.alignment_functions import linear_cca, to_latent_cca
        W1, W2, mu1, mu2 = linear_cca(source, target, 300)
        source_latents = to_latent_cca(en_dictionary.embed, W1, mu1)

    del source, target
    print("Done transform")

    n_words, n_dim = source_latents.shape

    outpath = 'data/wiki.{0}_{1}_aligned.txt'.format(src, tgt)
    fout = open(outpath, "w")

    # Header takes the guesswork out of loading by recording how many lines, vector dims
    fout.write(str(n_words) + " " + str(n_dim) + "\n")
    from tqdm import tqdm
    for i, token in enumerate(tqdm(source_latents)):
        vector_components = ["%.6f" % number for number in token]
        vector_as_string = " ".join(vector_components)

        out_line = en_dictionary.id2word[i] + " " + vector_as_string + "\n"
        fout.write(out_line)

    fout.close()


if __name__ == "__main__" :
    main()
