import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import logsumexp
import argparse
from tqdm import tqdm
# Code slightly modified from http://clic.cimec.unitn.it/~georgiana.dinu/down/
# Dinu, Georgiana, et al. "Improving zero-shot learning by mitigating
# https://arxiv.org/pdf/1412.6568.pdf

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def prec_at(ranks, cut):
    """
    computes precision score at a given rank
    """
    return len([r for r in ranks if r <= cut]) / float(len(ranks))


def get_rank(nn, gold):
    """
    Finds the rank for the provided translation column

    :param nn[ndarray 1xd]: vector of distance metrics of a specific
                            word (correspnding with items in gold)
                            with the target translation vocabulary

    :param gold [list[str]]: list of correct gold standard translations for
                             corresponding word

    :return [int]: index correponding to rank of translation
    """
    for idx,word in enumerate(nn):
        if word in gold:
            return idx + 1
    return idx + 1


def read_dict(dict_file):
    """
    :param dict_file[OrderedDict]: train/test dictionary from Dinu

    :param test_dict[OrderedDict]: language pair dict < source, target >
    """
    return  [tuple(line.strip().split()) for line in open(dict_file)]


def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    bilingual_dictionary = (bilingual_dictionary.items() if
                           not isinstance(bilingual_dictionary, list)
                           else bilingual_dictionary)

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)


def get_valid_data(source_dict, target_dict, test_dict):
    """
    Prepare test set

    :param source_dict[OrderedDict/FastVector]: dictionary for source language
    :param target_dict[OrderedDict/FastVector]: dictionary for source language
    :param test_dict[OrderedDict/FastVector]: dictionary for source language


    :return [OrderedDict/Fasttext]: pairs at the intersection of all 3 dictionaries

    """
    test_dict = test_dict.items() if not isinstance(test_dict, list) else test_dict
    return [(el1, el2) for el1,el2 in test_dict if
            el1 in source_dict and el2 in target_dict]


# from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalize(a, axis=-1, order=2):
    """
    Utility function to normalize the rows of a numpy array.

    :param a[nxd ndarray]: input matrix to normalise
    :param axis[int]: axis along whcih to perform the norm
    :param order[int]: p in L_p norm

    :return [nxd ndarray]: a with unit norm rows
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def softmax(S, axis=1, beta=1.0):
    """
    stable softmax
    :param S[nxm ndarray]: matrix of logits to which apply softmax to (S_ij matrix in Nils)
    :param beta[float]: temperature parameter from Nils

    :returns [nxm  ndarray]: softmaxed A
    """
    S_tilde = beta * S # Applying temperature ?
    denominator = logsumexp(S_tilde, axis)[:,None]
    return np.exp(S_tilde - denominator)


def score(sp1, sp2, gold, additional, soft=False):
    """
    Score for translating / aligning
    Code slightly modified from http://clic.cimec.unitn.it/~georgiana.dinu/down/
    Dinu, Georgiana, et al. "Improving zero-shot learning by mitigating

    :param sp1 [FastVector]: source language
    :param sp2 [FastVector]: target language
    :param gold [dict]: dictionary of ground truth translations from test set

    :param additional [bool]: modifies the ranking in a way I dont yet understand

    :return void: prints scores
    """

    print("Computing cosines and sorting target space elements")
    softmax = softmax if soft else lambda x: x
    sim_mat = softmax(cdist(sp2.embed, sp1.embed, metric="cosine"))

    print("Done computing distances")

    if additional:
        #for each element, computes its rank in the ranked list of
        #similarites. sorting done on the opposite axis (inverse querying)
        srtd_idx = np.argsort(np.argsort(sim_mat, axis=1), axis=1)

        #for each element, the resulting rank is combined with cosine scores.
        #the effect will be of breaking the ties, because cosines are smaller
        #than 1. sorting done on the standard axis (regular NN querying)
        srtd_idx = np.argsort(srtd_idx + sim_mat, axis=0)
    else:
        srtd_idx = np.argsort(sim_mat, axis=0)
        # import pdb; pdb.set_trace()

    ranks = []
    for i,el1 in enumerate(gold.keys()):

        sp1_idx = sp1.word2id[el1]
        # print(sp1_idx
        #print(the top 5 translations
        translations = []
        for j in range(5):
            sp2_idx = srtd_idx[j, sp1_idx]
            word, score = sp2.id2word[sp2_idx], -sim_mat[sp2_idx, sp1_idx]
            translations.append("\t\t%s:%.3f" % (word, score))

        translations = "\n".join(translations)

        #get the rank of the (highest-ranked) translation
        # import pdb; pdb.set_trace()
        rnk = get_rank(srtd_idx[:,sp1_idx].ravel(),
                       [sp2.word2id[el] for el in gold[el1]])
        ranks.append(rnk)

        # print(("\nId: %d Source: %s \n\tTranslation:\n%s \n\tGold: %s \n\tRank: %d" %
        #        (len(ranks), el1, translations, gold[el1], rnk))

    print("Corrected: %s" % str(additional))
    if additional:
        print("Total extra elements, Test(%d) + Additional:%d" % (len(gold.keys()),
                                                           sp1.embed.shape[0]))

    for k in [1,5,10, 100]:
        print("Prec@%d: %.3f" % (k, prec_at(ranks, k)))


def compute_csls_accuracy(x_src, x_tgt, lexicon,
                          lexicon_size=-1, k=10, bsz=1024, csls=True):
    """
    CSLS method to overcome hubness by FAIR
    (Conneau, Alexis, et al. "Word translation without parallel data."
     arXiv preprint((2017). https://arxiv.org/pdf/1710.04087.pdf) > This implementation
     Actually follows the lost in translation paper and not the Muse one

    :param x_src[FastVector]: source langauge matrix
    :param x_tgt[FastVector]: target langauge matrix
    :param lexicon[dict[list]]: keys are source lexicon and values are correct translations (test set)
                                keys and elements in the the lists should be integer indexes corresponding to the
                                position of the word in the design matrix (word2id)
    :param k[int]: number of nearest neighbours used in CSLS
    :param csls[bool]: will use NN if false

    :return [int]: accuracy@1 for translation
    """

    x_src_vec1 =  x_src.embed
    x_trgt_vec1 =  x_tgt.embed
    x_src_vec = x_src_vec1 / np.linalg.norm(x_src_vec1, axis=1)[:, np.newaxis] + 1e-8
    x_trgt_vec = x_trgt_vec1 /np.linalg.norm(x_trgt_vec1, axis=1)[:, np.newaxis] + 1e-8

    sc = np.dot(x_trgt_vec, x_src_vec.T)

    similarities = 2 * sc
    sc2 = np.zeros(x_trgt_vec.shape[0])

    if csls:
        for i in tqdm(range(0, x_trgt_vec.shape[0], bsz)):
            j = min(i + bsz, x_trgt_vec.shape[0])
            sc_batch = np.dot(x_trgt_vec[i:j, :], x_src_vec.T)
            dotprod = np.partition(sc_batch, -k, axis=1)[:, -k:]
            sc2[i:j] = np.mean(dotprod, axis=1)
        similarities -= sc2[:, np.newaxis]

    nn = np.argsort(-similarities, axis=0)
    print("Done with argsort")

    ranks = [None for i in lexicon]
    for i, k in enumerate(tqdm(lexicon.keys())):
        sp1_idx = x_src.word2id[k]

        preds =  nn[:,sp1_idx].ravel()
        truth = [x_tgt.word2id[el] for el in lexicon[k]]

        ranks[i] = get_rank(preds, truth)

    print("Done getting ranks")

    for k in [1,5,10]:
        print("Prec@%d: %.3f" % (k, prec_at(ranks, k)))
