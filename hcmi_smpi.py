import logging
import pymit
import sys
import numpy
import smpi

#MADELON_TRAIN = "./MADELON_FLOAT/madelon_train.data"
#MADELON_TRAIN_LABELS = "./MADELON_FLOAT/madelon_train.labels"

PATH = sys.argv[1]
max_features = int(sys.argv[2])

MADELON_TRAIN = PATH + "/madelon_train.data"
MADELON_TRAIN_LABELS = PATH + "/madelon_train.labels"

bins = 10


@smpi.collect(smpi.dist_type.broadcast, smpi.dist_type.broadcast)
@smpi.root
def load_madelon_data():

    data_raw = numpy.loadtxt(MADELON_TRAIN, dtype=numpy.float)
    labels = numpy.loadtxt(MADELON_TRAIN_LABELS, dtype=numpy.float)

    [num_examples, num_features] = data_raw.shape
    data_discrete = numpy.zeros([num_examples, num_features])
    for i in range(num_features):
        _, bin_edges = pymit._lib.histogram(data_raw[:, i], bins=bins)
        data_discrete[:, i] = pymit._lib.digitize(data_raw[:, i], bin_edges, right=False)

    X = data_discrete
    Y = labels
    return [X,Y]
    
@smpi.collect(smpi.dist_type.gather)
@smpi.distribute(smpi.dist_type.local, smpi.dist_type.local, smpi.dist_type.scatter)
def calculate_mi(X, Y, features):
    MI = numpy.full([len(features)], numpy.nan, dtype=numpy.float)
    for i,X_i in enumerate(features):
        MI[i] = pymit.I(X[:, X_i], Y , bins=[bins, 2])
    return [MI]

@smpi.collect(smpi.dist_type.gather)
@smpi.distribute(smpi.dist_type.local, smpi.dist_type.local, smpi.dist_type.scatter, smpi.dist_type.broadcast)
def calculate_jmi(X, Y, features, selected_features):
    JMI = numpy.full([len(features)], numpy.nan, dtype=numpy.float)

    for i,X_k in enumerate(features):
        if X_k in selected_features:
            continue
        jmi = 0
        for X_j in selected_features:
            sum1 = pymit.I(X[:, X_j], Y, bins=[bins, 2])
            sum2 = pymit.I_cond(X[:, X_k], Y, X[:, X_j], bins=[bins, 2, bins])
            jmi += sum1 + sum2
        JMI[i] = jmi
      
    return [JMI]

@smpi.root
def estimate_and_print(selected_features, metric):
    f = numpy.nanargmax(metric)
    selected_features.append(f)
    print("   {:>3d},  {:>3d}".format(len(selected_features), f))
    return selected_features

@smpi.root
def assert_correct_result(selected_features, max_features):
    expected_features = [241, 338, 378, 105, 472, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]
    if expected_features[:max_features] == selected_features:
        print("correct result computed")
    else:
        print("wrong result, or more than 20 features computed")

def main():
    X,Y = load_madelon_data()
    [_, n_features] = X.shape
    features = numpy.asarray(range(n_features))
    [MI] = calculate_mi(X, Y, features)
    selected_features = estimate_and_print([], MI)

    for i in range(1, max_features):
        [JMI] = calculate_jmi(X, Y, features, selected_features)
        selected_features = estimate_and_print(selected_features, JMI)
    
    assert_correct_result(selected_features, max_features)

if __name__ == "__main__":
    main()
