import argparse
import sys
sys.path.append("../src")
from model import UnetModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet model runner.')
    parser.add_argument('--pretrained', type=str, help='start with pretrained weights')
    parser.add_argument('--weights', type=str, help='file save weights to')
    parser.add_argument('--size', type=int, help='image size')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--fold', type=int, help='fold number (1 to 6)')

    args = parser.parse_args()

    pretrained = args.pretrained is not None
    batch_size = args.batch
    size = (args.size, args.size)
    batches = 16 * 100 // batch_size
    model = UnetModel(args.weights, input_size=size, patience=10)
    if pretrained:
        model.unet.load_weights(f"../weights/{args.pretrained}.best.weights.h5py")
    print(f"Size = {size}, batch size = {batch_size}")
    if pretrained:
        learn_rates = [0.00001, 0.000001]
        epochs_list = [50, 50]
    else:
        learn_rates = [0.0001, 0.00001, 0.000001]
        epochs_list = [50, 50, 50]
    for lr, epochs in zip(learn_rates, epochs_list):
        train_loss, val_loss = model.fit_resized(lr, epochs, args.fold, batch_size=batch_size,
                                                 batches=batches)
