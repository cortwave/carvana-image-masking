import os
import logging

import h5py
import numpy as np
from skimage.io import imread

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

MAIN_DIR = os.path.abspath("../data")
TRAIN_DIR = os.path.join(MAIN_DIR, 'train')
TEST_DIR = os.path.join(MAIN_DIR, 'test')
MASK_DIR = os.path.join(MAIN_DIR, 'train_masks')
TRAIN_FILE = os.path.join(MAIN_DIR, "train.h5")
TEST_FILE = os.path.join(MAIN_DIR, "test.h5")


class Dataset:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    @staticmethod
    def read_img(fname):
        return (imread(fname) / 255).astype(np.float16)

    def cache_train(self):
        logger.info('Creating cache file for train')
        file = h5py.File(TRAIN_FILE, 'w')
        train_files = os.listdir(TRAIN_DIR)
        x_data = file.create_dataset('x_data', shape=(len(train_files), 1280, 1918, 3), dtype=np.float16)
        y_data = file.create_dataset('y_data', shape=(len(train_files), 1280, 1918, 1), dtype=np.int8)
        names = file.create_dataset('names', shape=(len(train_files),), dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {len(train_files)} files in train')
        for i, fn in enumerate(os.listdir(TRAIN_DIR)):
            img = self.read_img(os.path.join(TRAIN_DIR, fn))
            x_data[i, :, :, :] = img
            y_data[i, :, :, :] = imread(os.path.join(MASK_DIR, fn.replace('.jpg', '_mask.gif'))).reshape(1280, 1918, 1)
            names[i] = fn
        file.close()

    def cache_test(self):
        logger.info('Creating cache file for test')
        file = h5py.File(TEST_FILE, 'w')
        test_files = os.listdir(TEST_DIR)
        x_data = file.create_dataset('x_data', shape=(len(test_files), 1280, 1918, 3), dtype=np.float16)
        names = file.create_dataset('names', shape=(len(test_files),), dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {len(test_files)} files in test')
        for i, fn in enumerate(os.listdir(TRAIN_DIR)):
            img = self.read_img(os.path.join(TRAIN_DIR, fn))
            x_data[i, :, :, :] = img
            names[i] = fn
        file.close()

    def cache(self):
        self.cache_train()
        self.cache_test()


if __name__ == '__main__':
    Dataset().cache()
