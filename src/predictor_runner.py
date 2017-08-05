import argparse
import sys
sys.path.append("../src")
from model import UnetModel
from predictor import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet model runner.')
    parser.add_argument('--weights', type=str, help='wiights for model initialization')
    parser.add_argument('--size', type=int, help='image size')

    args = parser.parse_args()

    weights = args.weights
    size = args.size
    img_size = (size, size)
    model = UnetModel("prediction", input_size=img_size)
    model.load_weights(f"../weights/{weights}.best.weights.h5py")
    predictor = Predictor(model.unet)
    predictor.create_resized_submission(img_size, threshold=0.25)
