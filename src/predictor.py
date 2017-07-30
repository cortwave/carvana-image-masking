from constants import HEIGHT, WIDTH
import numpy as np
import pandas as pd
from rle_encoder import rle_encode
from skimage.io import imread
from tqdm import tqdm
from joblib import Parallel, delayed


def predict_by_crops(model, img, crop_size=224):
    img = img / 255 - 0.5
    mask = np.zeros((HEIGHT, WIDTH))
    for h in range(0, HEIGHT - crop_size, crop_size):
        for w in range(0, WIDTH - crop_size, crop_size):
            crop = img[h:h + crop_size, w:w + crop_size, :]
            pred = model.predict(np.array([crop]))[0].squeeze()
            mask[h: h + crop_size, w: w + crop_size] = pred
    return mask


def predict_image(model, image_name, threshold, crop_size):
    image = imread(f"../data/test/{image_name}")
    mask = predict_by_crops(model, image, crop_size)
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


