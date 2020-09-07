import cv2 as cv
from glob import glob
import numpy as np


def initial_spilt(save=False):
    frogs = [(cv.imread(y), 0) for y in glob('data/frogs/frog_*.png')]
    toads = [(cv.imread(y), 1) for y in glob('data/toads/toad_*.png')]
    dset = frogs + toads
    np.random.shuffle(dset)
    X = np.array([x[0] for x in dset])
    Y = np.array([y[1] for y in dset])
    if not save:
        return X, Y
    else:
        i = 0
        for img, isToad in dset:
            img = cv.resize(img, (128, 128))
            cv.imwrite(f'data/dataset/{(i := i + 1)}_{isToad}.png', img)


#  196 test, 220 train and dev

def split_train_test(X, Y, num_folds=5, test_size=196):
    fold_size = (X.shape[0] - test_size) // num_folds
    fold_inds = [fold_size * i for i in range(1, num_folds + 1)]
    *train_folds_X, test_X = np.split(X, fold_inds)
    *train_folds_Y, test_Y = np.split(Y, fold_inds)
    return (train_folds_X, train_folds_Y), (test_X, test_Y)
