#!/usr/bin/env python
import os
import sys
import glob
import shutil

""" Run this script from the base directory of the project.
Assumes that the distributed conll-2012 data directory is
one level above the base directory.
"""

def main(dataset):
    """dataset is either 'train' or 'test'
    """
    if dataset == 'train':
        data_dir = os.path.abspath('../conll-2012/train/english/annotations/')
        new_dir = os.path.abspath('../data/train')
        train_files = glob.glob(os.path.join(data_dir, '*/*/*/*auto_conll'))

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for f in train_files:
            shutil.copyfile(f, os.path.join(new_dir, os.path.basename(f)))

        print('Copied {} files from {} to {}'.format(
            len(train_files), data_dir, new_dir))
    else:
        data_dir = os.path.abspath('../conll-2012/test/english/annotations/')
        new_dir = os.path.abspath('../data/test')
        test_files = glob.glob(os.path.join(data_dir, '*/*/*/*gold_conll'))

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for f in test_files:
            shutil.copyfile(f, os.path.join(new_dir, os.path.basename(f)))

if __name__ == '__main__':
    main(sys.argv[1])
