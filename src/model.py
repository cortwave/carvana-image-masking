from generator import train_generator, valid_generator, train_generator_resized, valid_generator_resized
from callbacks import LossHistory
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, Adam
from dice import dice_coef, dice_coef_loss, dice_bce_loss
from unet import get_unet


class UnetModel:
    def __init__(self, model_name, input_size=(224,224), patience=20):
        self.input_size = input_size
        self.unet = get_unet(input_size=input_size)
        self.callbacks = [LossHistory(),
                          ModelCheckpoint(f"../weights/{model_name}.best.weights.h5py", save_best_only=True, verbose=1, monitor="val_dice_coef", mode='max'),
                          EarlyStopping(monitor="val_dice_coef", patience=patience, mode='max')]

    def _get_losses(self):
        return self.callbacks[0].train_losses, self.callbacks[0].val_losses

    def load_weights(self, name):
        self.unet.load_weights(name)

    def predict(self, images):
        return self.unet.predict(images)

    def fit(self, lr, epochs, n_fold, batch_size=8, opt="Adam", batches=400):
        if opt == "Adam":
            optimizer = Adam(lr=lr)
        elif opt == "SGD":
            optimizer = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
        else:
            raise Exception(f"Unknown optimizer: {opt}")
        train_gen = train_generator(n_fold=n_fold, batch_size=batch_size)
        valid_gen = valid_generator(n_fold=n_fold, batch_size=batch_size)
        self.unet.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=[dice_coef])
        self.unet.fit_generator(train_gen,
                                steps_per_epoch=batches,
                                nb_epoch=epochs,
                                validation_data=valid_gen,
                                validation_steps=batches,
                                callbacks=self.callbacks)
        return self._get_losses()

    def fit_resized(self, lr, epochs, n_fold, batch_size=8, opt="Adam", batches=400):
        if opt == "Adam":
            optimizer = Adam(lr=lr)
        elif opt == "SGD":
            optimizer = SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True)
        else:
            raise Exception(f"Unknown optimizer: {opt}")
        train_gen = train_generator_resized(n_fold=n_fold, batch_size=batch_size, size=self.input_size)
        valid_gen = valid_generator_resized(n_fold=n_fold, batch_size=batch_size, size=self.input_size)
        self.unet.compile(optimizer=optimizer, loss=dice_bce_loss, metrics=[dice_coef])
        self.unet.fit_generator(train_gen,
                                steps_per_epoch=batches,
                                nb_epoch=epochs,
                                validation_data=valid_gen,
                                validation_steps=batches // 2,
                                callbacks=self.callbacks)
        return self._get_losses()


