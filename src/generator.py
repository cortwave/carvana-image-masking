import numpy as np
import h5py
import random
from constants import HEIGHT, WIDTH
from skimage.transform import resize
from skimage.io import imread
import cv2
from scipy import ndimage
import os

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
            if is_train:
                img, mask = augmentate(img, mask)
            images.append(img)
            masks.append(mask)
        yield preprocess_batch(np.array(images)), np.array(masks)


def augmentate(img, mask):
    if random.random() < 0.25:
        flipcode = random.choice([-1, 0, 1])
        img = cv2.flip(img, flipcode)
        mask = np.expand_dims(cv2.flip(mask, flipcode), axis=3)
    if random.random() < 0.25:
        angle = random.choice([90, 180, 270])
        img = ndimage.rotate(img, angle)
        mask = np.expand_dims(ndimage.rotate(mask.squeeze(), angle), axis=3)
    return img, mask


def train_generator(n_fold, batch_size=32, crop_size=224):
    return generator(n_fold, True, batch_size, crop_size)


def valid_generator(n_fold, batch_size=32, crop_size=224):
    return generator(n_fold, False, batch_size, crop_size)


def generator(n_fold, is_train, batch_size, crop_size):
    folder = "../data/train"
    pred_folder = "../data/train_pred"
    mask_folder = "../data/train_masks"
    files = sorted(os.listdir(folder))
    while True:
        images = []
        masks = []
        index = get_train_index(n_fold) if is_train else get_valid_index(n_fold)
        image_name = files[index]
        image = imread(f"{folder}/{image_name}")
        pred_mask = imread(f"{pred_folder}/{image_name}")
        mask = imread(f"{mask_folder}/{image_name.split('.')[0]}_mask.gif")
        for _ in range(batch_size):
            h, w = find_crop(mask, crop_size)
            img = np.dstack((image[h:h + crop_size, w:w + crop_size, :], pred_mask[h:h + crop_size, w:w + crop_size]))
            msk = mask[h:h + crop_size, w:w + crop_size]
            img, msk = augmentate(img, msk)
            msk = np.expand_dims(msk.squeeze(), axis=3)
            images.append(img)
            masks.append(msk)
        yield preprocess_batch(np.array(images)), np.array(masks) / 256


def find_crop(mask, crop_size):
    while True:
        h = random.randint(0, HEIGHT - crop_size)
        w = random.randint(0, WIDTH - crop_size)
        img_crop = mask[h:h + crop_size, w:w + crop_size]
        mean = img_crop.mean()
        if not 50 < mean < 200:
            continue
        else:
            return h, w


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