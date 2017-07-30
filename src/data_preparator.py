import os
import logging

import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from constants import HEIGHT, WIDTH

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

MAIN_DIR = os.path.abspath("data")
TRAIN_DIR = os.path.join(MAIN_DIR, 'train')
TEST_DIR = os.path.join(MAIN_DIR, 'test')
MASK_DIR = os.path.join(MAIN_DIR, 'train_masks')
TRAIN_FILE = os.path.join(MAIN_DIR, "train.h5")
TEST_FILE = os.path.join(MAIN_DIR, "test.h5")


class Dataset:
    def __init__(self):
        pass

    @staticmethod
    def read_img(fname):
        return imread(fname).astype(np.uint8)

    def cache_train(self):
        logger.info('Creating cache file for train')
        train_files = sorted(os.listdir(TRAIN_DIR))
        train_size = len(train_files)
        file = h5py.File(TRAIN_FILE, 'w')
        x_data = file.create_dataset('x_data', shape=(train_size, HEIGHT, WIDTH, 3), dtype=np.uint8)
        y_data = file.create_dataset('y_data', shape=(train_size, HEIGHT, WIDTH, 1), dtype=np.uint8)
        names = file.create_dataset('names', shape=(train_size,), dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {train_size} files in train')
        for i, fn in tqdm(enumerate(train_files), total=train_size):
            img = self.read_img(os.path.join(TRAIN_DIR, fn))
            x_data[i, :, :, :] = img
            y_data[i, :, :, :] = imread(os.path.join(MASK_DIR, fn.replace('.jpg', '_mask.gif'))).reshape(HEIGHT, WIDTH, 1)
            names[i] = fn
        file.close()

    def cache_test(self):
        logger.info('Creating cache file for test')
        file = h5py.File(TEST_FILE, 'w')
        test_files = sorted(os.listdir(TEST_DIR))
        test_size = len(test_files)
        x_data = file.create_dataset('x_data', shape=(test_size, HEIGHT, WIDTH, 3), dtype=np.uint8)
        names = file.create_dataset('names', shape=(test_size,), dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {test_size} files in test')
        for i, fn in tqdm(enumerate(test_files), total=test_size):
            img = self.read_img(os.path.join(TEST_DIR, fn))
            x_data[i, :, :, :] = img
            names[i] = fn
        file.close()

    def cache(self):
        self.cache_train()
        self.cache_test()


if __name__ == '__main__':
    Dataset().cache()
