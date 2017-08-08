from model import UnetModel
from skimage.io import imread, imsave
from skimage.transform import resize
from constants import HEIGHT, WIDTH
import os
from tqdm import tqdm
import numpy as np

def predict(file, save_to, model, size):
    img = imread(file)
    img = resize(img, size, mode='constant') / 255
    pred = model.predict(np.array([img]))[0]
    pred = resize(pred, (HEIGHT, WIDTH), mode='constant')
    imsave(save_to, (pred.squeeze() * 255).astype('uint8'))

def predict_train(model, size):
    print("train prediction")
    files = os.listdir("../data/train")
    for file in tqdm(files):
        predict(f"../data/train/{file}", f"../data/train_pred/{file}", model, size)


def predict_test(model, size):
    print("test prediction")
    files = os.listdir("../data/test")
    for file in tqdm(files):
        predict(f"../data/test/{file}", f"../data/test_pred/{file}", model, size)


if __name__ == '__main__':
    model = UnetModel("Mask prediction", input_size=(1024, 1024))
    model.load_weights("../weights/1024.best.weights.h5py")
    predict_train(model, (1024, 1024))
    predict_test(model, (1024, 1024))




