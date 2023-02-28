import numpy as np
from glob import glob

images = glob('/lscratch/jacaraba/ethiopia-lcluc/landcover.v1/images/*.npy')
labels = glob('/lscratch/jacaraba/ethiopia-lcluc/landcover.v1/labels/*.npy')

for i in images:
    x = np.load(i)
    print(x.shape, x.min(), x.max())
    if x.min() < 0:
        print(i, "x < 0", x.min())

for i in labels:
    x = np.load(i)
    print(x.shape, x.min(), x.max())
    if x.min() < 0:
        print(i, "x < 0", x.min())

