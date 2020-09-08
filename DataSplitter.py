import cv2 as cv
import numpy as np
from glob import glob


def parse_data(*locations, extensions='.png', save=False, save_loc='', size=(128, 128), seed=0):

    """ Parses all files from locations with proper extension and generates a
    database in random order with flattened images of entered size. """

    np.random.seed(seed)
    data = []
    for i, loc in enumerate(locations):
        for ext in extensions:
            data += [(cv.resize(cv.imread(y), size), i) for y in glob(f'{loc}/*{ext}')]

    np.random.shuffle(data)

    X = np.array([x[0].flatten() for x in data])
    Y = np.array([x[1] for x in data])

    if save:
        i = 0
        for img, ind in data:
            cv.imwrite(f'{save_loc}/{(i := i + 1)}_{ind}.png', img)
    return X, Y


#  196 test, 220 train and dev

def split_train_test(X, Y, num_folds=5, test_size=196):

    """ Splits the data in several folds to perform cross-validation. """

    fold_size = (X.shape[0] - test_size) // num_folds
    fold_inds = [fold_size * i for i in range(1, num_folds + 1)]
    *train_folds_X, test_X = np.split(X, fold_inds)
    *train_folds_Y, test_Y = np.split(Y, fold_inds)
    return train_folds_X, train_folds_Y, test_X, test_Y


if __name__ == '__main__':
    folds_X, folds_Y, *_ = split_train_test(*parse_data('data/frogs', 'data/toads', save=False))
    print([x.shape for x in folds_X])
    print([y[0] for y in folds_Y])
