import cv2 as cv
from glob import glob
from os import remove

images = ((y, cv.imread(y, cv.IMREAD_UNCHANGED)) for y in glob('data/samples/*.png'))

with open('data/counters.txt', 'r') as r:
    frog_ind, toad_ind = map(int, r.read().split())

for (name, img) in images:

    cv.imshow('Tinder', img)

    key = cv.waitKey(0)

    if key & 0xFF == ord('q'):
        remove(name)
        cv.imwrite(f'data/frogs/frog_{(frog_ind := frog_ind + 1)}.png', img)
    elif key & 0xFF == ord('e'):
        remove(name)
        cv.imwrite(f'data/toads/toad_{(toad_ind := toad_ind + 1)}.png', img)
    elif key & 0xFF == ord('d'):
        remove(name)
    elif key & 0xFF == ord('s'):
        with open('data/counters.txt', 'w') as w:
            w.write(f'{frog_ind}\n{toad_ind}')
        break
