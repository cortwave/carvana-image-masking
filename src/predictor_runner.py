import argparse
import sys
sys.path.append("../src")
from model import UnetModel
from predictor import Predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet model runner.')
    parser.add_argument('--weights', type=str, help='weights for model initialization')
    parser.add_argument('--size', type=int, help='image size')
    parser.add_argument('--mode', type=str, help='prediction mode')

    args = parser.parse_args()

    weights = args.weights
    size = args.size
    img_size = (size, size)
    input_channels = 4 if args.mode == 'crop' else 3
    model = UnetModel("prediction", input_size=img_size, input_channels=input_channels)
    model.unet.load_weights(f"../weights/{weights}.best.weights.h5py")
    predictor = Predictor(model.unet)
    if args.mode == 'resized':
        predictor.create_resized_submission(img_size, threshold=0.25)
    if args.mode == 'crop':
        predictor.create_crop_submission(threshold=0.25)
