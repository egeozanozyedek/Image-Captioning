import numpy
import h5py
import toolbox

TRAIN_DATA_FILENAME = "eee443_project_dataset_train.h5"
f = h5py.File(TRAIN_DATA_FILENAME, "r")

data = {}

for key in list(f.keys()):
    data[key] = f[key][()]
    print(key, ":", data[key].shape)

