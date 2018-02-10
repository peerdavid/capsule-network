import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from keras import callbacks, layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

import utils
from capsule_layers import PrimaryCaps, DigitCaps, Length, Mask, margin_loss


#
# Set defaults
#
K.set_image_data_format('channels_last')


#
# Main
#
def main(args):
    # Ensure working dirs
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Create model
    model, eval_model, manipulate_model = create_model(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  num_routing=args.num_routing)
    model.summary()

    # Run training / testing
    if args.weights is not None and os.path.exists(args.weights):
        model.load_weights(args.weights)
        print("Successfully loaded weights file %s" % args.weights)
    
    if not args.testing:
        print("\n" + "=" * 40 + " TRAIN " + "=" * 40)
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:
        print("\n" + "=" * 40 + " TEST =" + "=" * 40)
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)
    
    print("=" * 40 + "=======" + "=" * 40)


def load_mnist():
    # the data, shuffled and split between train and test sets
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


def create_model(input_shape, n_class, num_routing):
    # Create CapsNet
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primary_caps = PrimaryCaps(layer_input=conv1, name='PrimaryCaps', dim_capsule=8, channels=32, kernel_size=9, strides=2)
    digit_caps = DigitCaps(num_capsule=n_class, dim_vector=16, num_routing=num_routing)(primary_caps)
    out_caps = Length(name='CapsNet')(digit_caps)

    # Create decoder
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digit_caps, y])    # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digit_caps)              # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digit_caps = layers.Add()([digit_caps, noise])
    masked_noised_y = Mask()([noised_digit_caps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))

    return train_model, eval_model, manipulate_model


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_CapsNet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'CapsNet': 'accuracy'})

    def train_generator_with_augmentation(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    model.fit_generator(generator=train_generator_with_augmentation(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint, lr_decay])


    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    utils.plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('Test accuracy:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = utils.combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, args):
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = utils.combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('Manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))


#
# Main
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")

    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")

    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")

    parser.add_argument('-r', '--num_routing', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")

    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")

    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")

    parser.add_argument('--save_dir', default='./result')

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")

    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    main(args)