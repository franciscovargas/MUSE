from src.fasttext import FastVector
from src.dinu_experiments.svd import learn_transformation
from src.dinu_experiments.utils import (read_dict, get_valid_data, make_training_matrices,
                                        compute_csls_accuracy, str2bool)
from src.alignment_functions import guess_for_closed_form_fa, to_latent
from src.dinu_experiments.cca import linear_cca, to_latent_cca
from os.path import join
import numpy as np
import collections
import argparse
from pprint import PrettyPrinter

# This program reproduces experiments from Dinu 2014

data_path = "data"


def test_results_using_dinu_data(source_dict_location=join(data_path, 'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'),
                                 target_dict_location=join(data_path, 'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'),
                                 test_pairs_location=join(data_path, "crosslingual/dictionaries/OPUS_en_it_europarl_test.txt"),
                                 train_pairs_location=join(data_path, "crosslingual/dictionaries/OPUS_en_it_europarl_train_5K.txt"),
                                 swap=False, expert=False, method='pcca', csls=True):
    """
    :param source_dict[dict/FastVector]: word, vector key value pairs for source language
    :param target_dict[dict/FastVector]: word, vector key value pairs for target language
    :param test_pairs: list (source, target) ground truth test set pairs
    :param swap[bool]: when true do italian to english translation other wise do english to italian
    :param expert[bool]: if false will use the pseudo-dictionary otherwise will use the expert one form Dinu
    :param csls[bool]: if true will use csls retrieval from MUSE paper else will use NN

    :return void: just prints results
    """

    target_dict = FastVector(vector_file=target_dict_location)
    source_dict = FastVector(vector_file=source_dict_location)
    if swap:
        target_dict, source_dict = source_dict, target_dict

    if not expert:
        source_words = set(source_dict.word2id.keys())
        target_words = set(target_dict.word2id.keys())
        overlap = list(source_words & target_words)
        train_dict = zip(overlap, overlap)
    else:
        train_dict = read_dict(train_pairs_location)

    test_dict = read_dict(test_pairs_location)
    if swap:
        test_dict = [(y, x) for (x, y) in test_dict]
        train_dict = [(y, x) for (x, y) in train_dict]

    # Obtain alignment matrices
    source_matrix_train, target_matrix_train = make_training_matrices(
        source_dict, target_dict, train_dict
    )

    # extract valid data (intersection in all 3 sets)
    test_dict = get_valid_data(source_dict, target_dict, test_dict)

    # Mapping using Smith et al. (2017)
    if method == 'svd':
        transform, mean_x, mean_y = learn_transformation(source_matrix_train,
                                                         target_matrix_train,
                                                         svd_mean=svd_mean)
        source_dict.apply_transform(transform)

    ### SKLearn CCA (NIPALS)
    if method == 'cca_sklearn':
        from sklearn.cross_decomposition import CCA

        cca = CCA(n_components=300)
        cca.fit(source_matrix_train, target_matrix_train)

        source_latent = cca.transform(source_dict.embed)
        _, target_latent = cca.transform(np.zeros_like(target_dict.embed), target_dict.embed)

    # PCCA (Our method)
    if method == 'pcca':
        Ws, Psis, mus, U1, U2, P_sqrt = guess_for_closed_form_fa([source_matrix_train, target_matrix_train], return_components=True)

        source_latent = to_latent(source_dict.embed, Ws[0], Psis[0], mus[0])
        target_latent = to_latent(target_dict.embed, Ws[1], Psis[1], mus[1])

    # ### CCA
    if method == "cca":
        W1, W2, m1, m2 = linear_cca(source_matrix_train, target_matrix_train, 300)

        source_latent = to_latent_cca(source_dict.embed, W1, m1)
        target_latent = to_latent_cca(target_dict.embed, W2, m2)

    if method != "svd":
        source_dict.set_embeddings(source_latent)
        target_dict.set_embeddings(target_latent)

    # turn test data into a dictionary (a word can have mutiple translations)
    gold = collections.defaultdict(set)
    gold2 = collections.defaultdict(set)
    for k, v in test_dict:
        gold[k].add(v)
        gold2[source_dict.word2id[k]].add(target_dict.word2id[v])

    compute_csls_accuracy(source_dict.subset([k for k, v in test_dict]),
                          target_dict, gold, len(gold), csls=csls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experriemnts for pcca')
    parser.add_argument('--swap', type=str2bool, default=False,
                        help='swap the target and source languages (do it-en instead)')
    parser.add_argument('--expert', type=str2bool, default=True,
                        help='If true use expert dict else use pseudo_dict')
    parser.add_argument('--method', default="pcca", choices=["pcca", "svd", "cca", "cca_sklearn"],
                        help='Select method from pcca|svd|cca|cca_sklearn to run results on')
    parser.add_argument('--source_dict_location', default="data/EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt",
                        help='Source dict location')
    parser.add_argument('--target_dict_location', default="data/IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt",
                        help='Target dict location')
    parser.add_argument('--csls', default=False, type=str2bool,
                        help='If true use csls retrieval otherwise use regular NN')

    args = parser.parse_args()

    # Print args
    PrettyPrinter(indent=2).pprint(vars(args))

    test_results_using_dinu_data(source_dict_location=args.source_dict_location,
                                 target_dict_location=args.target_dict_location,
                                 swap=args.swap,
                                 expert=args.expert,
                                 method=args.method,
                                 csls=args.csls)
