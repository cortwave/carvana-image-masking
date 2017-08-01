import numpy as np
import h5py
import random
from constants import HEIGHT, WIDTH
from skimage.transform import resize

TRAIN_SIZE = 318 * 16
FOLDS_COUNT = 6
FOLD_SIZE = int(TRAIN_SIZE / FOLDS_COUNT)


def preprocess_batch(batch):
    return batch / 256


def train_generator_resized(n_fold, size=(224, 224), batch_size=32):
    return resized_generator(n_fold, True, batch_size, size)


def valid_generator_resized(n_fold, size=(224, 224), batch_size=32):
    return resized_generator(n_fold, False, batch_size, size)


def resized_generator(n_fold, is_train, batch_size, size):
    train = h5py.File("../data/train.h5")
    x_data = train["x_data"]
    y_data = train["y_data"]
    while True:
        images = []
        masks = []
        for _ in range(batch_size):
            index = get_train_index(n_fold) if is_train else get_valid_index(n_fold)
            img = resize(x_data[index], size, mode="constant")
            mask = resize(y_data[index], size, mode="constant")
            images.append(img)
            masks.append(mask)
        yield preprocess_batch(np.array(images)), np.array(masks)


def train_generator(n_fold, batch_size=32):
    return generator(n_fold, True, batch_size)


def valid_generator(n_fold, batch_size=32):
    return generator(n_fold, False, batch_size)


def generator(n_fold, is_train, batch_size):
    train = h5py.File("../data/train.h5")
    x_data = train["x_data"]
    y_data = train["y_data"]
    while True:
        index = get_train_index(n_fold) if is_train else get_valid_index(n_fold)
        img, mask = x_data[index], y_data[index]
        images = []
        masks = []
        for _ in range(batch_size):
            rand_img, rand_mask = random_crop(img, mask=mask)
            images.append(rand_img)
            masks.append(rand_mask)
        yield preprocess_batch(np.array(images)), np.array(masks)


def get_train_index(n_fold):
    fold_start, fold_end = get_fold_boundaries(n_fold)
    index = random.randint(0, TRAIN_SIZE - 1)
    index = index + FOLD_SIZE if fold_start <= index < fold_end else index
    return index


def get_valid_index(n_fold):
    fold_start, fold_end = get_fold_boundaries(n_fold)
    index = random.randint(fold_start, fold_end)
    return index


def get_fold_boundaries(n_fold):
    fold_start = (n_fold - 1) * FOLD_SIZE
    fold_end = fold_start + FOLD_SIZE
    return fold_start, fold_end


def random_crop(img, crop_size=224, mask=None):
    h = random.randint(0, HEIGHT - crop_size)
    w = random.randint(0, WIDTH - crop_size)
    img_crop = img[h:h+crop_size, w:w+crop_size,:]
    if mask != None:
        mask_crop = mask[h:h+crop_size, w:w+crop_size,:]
        return img_crop, mask_crop
    else:
        return img_crop