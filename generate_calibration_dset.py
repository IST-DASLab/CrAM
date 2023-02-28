# From https://github.com/papers-submission/structured_transposable_masks

import os
import shutil

basepath = '/home/Datasets/ILSVRC/Data/CLS-LOC/train/'
basepath_calib = '/home/Datasets/ILSVRC/calib_1/'

directory = os.fsencode(basepath)
os.mkdir(basepath_calib)
for d in os.listdir(directory):
    dir_name = os.fsdecode(d)
    dir_path = os.path.join(basepath, dir_name)
    dir_copy_path = os.path.join(basepath_calib, dir_name)
    os.mkdir(dir_copy_path)
    for i, f in enumerate(os.listdir(dir_path)):
        if i == 1:
            break
        file_path = os.path.join(dir_path, f)
        copy_file_path = os.path.join(dir_copy_path, f)
        shutil.copyfile(file_path, copy_file_path)
