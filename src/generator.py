import numpy as np
import h5py

TRAIN_SIZE = 318 * 16
FOLDS_COUNT = 6
FOLD_SIZE = int(TRAIN_SIZE / FOLDS_COUNT)


def train_generator(n_fold, batch_size=32):
    train = h5py.File("../data/train.h5")
    x_data = train["x_data"]
    y_data = train["y_data"]
    while True:
        indexes = get_train_indexes(n_fold, batch_size)
        yield x_data[indexes], y_data[indexes]


def valid_generator(n_fold, batch_size=32):
    train = h5py.File("../data/train.h5")
    x_data = train["x_data"]
    y_data = train["y_data"]
    while True:
        indexes = get_valid_indexes(n_fold, batch_size)
        yield x_data[indexes], y_data[indexes]


def get_train_indexes(n_fold, batch_size):
    fold_start, fold_end = get_fold_boundaries(n_fold)
    indexes = np.random.choice(TRAIN_SIZE, batch_size, replace=False)
    indexes = np.array([i + FOLD_SIZE if fold_start <= i < fold_end else i for i in indexes])
    return sorted(list(indexes))


def get_valid_indexes(n_fold, batch_size):
    fold_start, fold_end = get_fold_boundaries(n_fold)
    indexes = np.random.choice(range(fold_start, fold_end), batch_size, replace=False)
    return sorted(list(indexes))


def get_fold_boundaries(n_fold):
    fold_start = (n_fold - 1) * FOLD_SIZE
    fold_end = fold_start + FOLD_SIZE
    return fold_start, fold_end