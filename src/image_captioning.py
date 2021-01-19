import numpy
import h5py
import constant
import toolbox


f = h5py.File(constant.TRAIN_DATA_FILENAME, "r")
print(constant.TRAIN_DATA_FILENAME)

data = {}

for key in list(f.keys()):
    data[key] = f[key][()]
    print(key, ":", data[key].shape)


