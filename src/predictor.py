from constants import HEIGHT, WIDTH
import numpy as np
import pandas as pd
from rle_encoder import rle_encode
from skimage.io import imread
from tqdm import tqdm
from generator import valid_generator_resized
from skimage.transform import resize

def predict_by_crops(model, img, crop_size=224):
    img = img / 255 - 0.5
    mask = np.zeros((HEIGHT, WIDTH))
    for h in range(0, HEIGHT - crop_size, crop_size):
        for w in range(0, WIDTH - crop_size, crop_size):
            crop = img[h:h + crop_size, w:w + crop_size, :]
            pred = model.predict(np.array([crop]))[0].squeeze()
            mask[h: h + crop_size, w: w + crop_size] = pred
    return mask


def predict_resized(model, img, size):
    img = resize(img, size, mode="constant")
    img = img / 256
    mask = model.predict(np.array([img]))[0]
    mask = resize(mask, (HEIGHT, WIDTH), mode="constant")
    return mask

def predict_image(model, image_name, threshold, crop_size):
    image = imread(f"../data/test/{image_name}")
    mask = predict_by_crops(model, image, crop_size)
    rle = rle_encode(mask.squeeze() > threshold)
    return rle

def predict_image_resized(model, image_name, threshold, size):
    image = imread(f"../data/test/{image_name}")
    mask = predict_resized(model, image, size)
    rle = rle_encode(mask.squeeze() > threshold)
    return rle

class Predictor:
    def __init__(self, model):
        self.model = model

    def create_crop_submission(self, crop_size=224, threshold=0.97):
        submission = pd.read_csv("../data/sample_submission.csv")
        rles = [predict_image(self.model, image_name, threshold, crop_size) for image_name in tqdm(submission['img'], total=submission.shape[0])]
        submission['rle_mask'] = np.array(rles)
        submission.to_csv("crop_submission.csv", index=False)

    def dice_coef(self, real, pred):
        all = []
        for r, p in zip(real, pred):
            all.append(np.sum(np.logical_and(r, p)) * 2.0 / (np.sum(p) + np.sum(r)))
        return np.mean(np.array(all))

    def find_threshold(self, size, n_fold):
        batch = next(valid_generator_resized(n_fold, size=size, batch_size=4))
        pred = self.model.predict(batch[0])
        real = batch[1]
        best_coef = 0
        best_t = 0.05
        for t in np.arange(0.05, 1.00, 0.05):
            print(t)
            coef = self.dice_coef(real, pred > t)
            print(coef)
            if coef > best_coef:
                print(f"New best threshold is {t} with score {coef}")
                best_coef = coef
                best_t = t
        return best_t

    def create_resized_submission(self, size, threshold=0.90):
        submission = pd.read_csv("../data/sample_submission.csv")
        rles = [predict_image_resized(self.model, image_name, threshold, size) for image_name in tqdm(submission['img'], total=submission.shape[0])]
        submission['rle_mask'] = np.array(rles)
        submission.to_csv("resized_submission.csv", index=False)


