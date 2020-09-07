import cv2 as cv
from glob import glob
from numpy.random import shuffle

frogs = [(cv.imread(y), 0) for y in glob('data/frogs/frog_*.png')]
toads = [(cv.imread(y), 1) for y in glob('data/toads/toad_*.png')]
dset = frogs + toads
shuffle(dset)

i = 0
for img, isToad in dset:
    img = cv.resize(img, (128, 128))
    cv.imwrite(f'data/dataset/{(i := i + 1)}_{isToad}.png', img)