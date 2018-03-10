import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import keras
from keras import callbacks, layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score

import foolbox
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability

import utils


#
# Set defaults
#
K.set_image_data_format('channels_last')
none_of_the_above_class = 1


#
# Main
#
def main(args):
    # Ensure working dirs
    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # Save args into file 
    if not args.testing:
        with open(args.save_dir+"/args.txt", "w") as out:
            sorted_args = sorted(vars(args).items())
            out.write('\n'.join("{0} = {1}".format(a, v) for (a, v) in sorted_args))

    # Set learning phase for tf
    if args.testing or args.fool:
        keras.backend.set_learning_phase(0)

    # Load data
    (x_train, y_train), (x_test, y_test), n_class = load_dataset(none_of_the_above_class)
    print("\nNum training samples: %d"% len(x_train))
    print("Num testing samples: %d\n" % len(x_test))

    # Cut off training samples
    if(args.max_num_samples is not None):
        x_train = x_train[:args.max_num_samples]
        y_train = y_train[:args.max_num_samples]
        print("\nUsing only %d training samples.\n" % len(x_train))

    # Calc shape depending on cropping 
    shape = (args.crop_x, args.crop_y, x_train.shape[1:3])  \
            if args.crop_x is not None and args.crop_y is not None \
            else x_train.shape[1:]

    # Create model
    model = create_convnet(input_shape=shape, n_class=n_class)
    model.summary()

    # Run training / testing
    if args.weights is not None and os.path.exists(args.weights):
        model.load_weights(args.weights)
        print("Successfully loaded weights file %s" % args.weights)
    else:
        print('(Warning) No weights are provided, using random initialized weights.')

    # Run test / fool / train
    if args.testing:
        print("\n" + "=" * 40 + " TEST =" + "=" * 40)
        test(model=model, data=(x_test, y_test), args=args)
    elif args.fool:
        print("\n" + "=" * 40 + " FOOL =" + "=" * 40)
        adversarial_attack(model, x_test, y_test)
    else:
        print("\n" + "=" * 40 + " TRAIN " + "=" * 40)
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

        
    
    print("=" * 40 + "=======" + "=" * 40)


def load_dataset(with_non_of_the_above_class):
    # the data, shuffled and split between train and test sets
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255

    # ... we also found that it helped to introduce a "none-of-the-above" category
    n_class = 10 + with_non_of_the_above_class
    y_train = to_categorical(y_train.astype('float32'), num_classes=n_class)
    y_test = to_categorical(y_test.astype('float32'), num_classes=n_class)

    return (x_train, y_train), (x_test, y_test), n_class


def create_convnet(input_shape, n_class):
    """ As described in [1] chapter 5
    """
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv2 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu', name='conv3')(conv2)
    flat = layers.Flatten()(conv3)
    fc1 = layers.Dense(328, activation='relu')(flat)
    fc2 = layers.Dense(192, activation='relu')(fc1)
    dp1 = layers.Dropout(0.5)(fc2)
    out = layers.Dense(n_class, activation='softmax')(dp1)

    model = models.Model(x, out)
    return model


def train(model, data, args):
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.hdf5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])      

    # Generator with data augmentation as used in [1] ([...] also trained on 2-pixel shifted MNIST)
    def train_generator_with_augmentation(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            if args.crop_x is not None and args.crop_y is not None:
                x_batch = utils.random_crop(x_batch, [args.crop_x, args.crop_y])  
            yield (x_batch, y_batch)

    
    generator = train_generator_with_augmentation(x_train, y_train, args.batch_size, args.shift_fraction)
    model.fit_generator(generator=generator,
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[x_test, y_test],
                        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.hdf5')
    print('Trained model saved to \'%s/trained_model.hdf5\'' % args.save_dir)

    utils.plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):

    # Create an augmentation function and cache augmented samples
    # to be displayed later
    def test_generator_with_augmentation(x, batch_size, shift_range, rotation_range):
        test_datagen = ImageDataGenerator(width_shift_range=shift_range,
                                          height_shift_range=shift_range,
                                          rotation_range=rotation_range)
        generator = test_datagen.flow(x, batch_size=batch_size, shuffle=False)
        while 1:
            x_batch = generator.next()
            if args.crop_x is not None and args.crop_y is not None:
                x_batch = utils.random_crop(x_batch, [args.crop_x, args.crop_y])
            yield (x_batch)


    # Run predictions
    test_batch_size = 100
    x_true, y_true = data
    generator = test_generator_with_augmentation(x_true, test_batch_size, args.shift_fraction, args.rotation_range)
    y_pred = model.predict_generator(generator=generator, steps=len(x_true) // test_batch_size)

    # Print different metrics using the top score
    y_true = np.argmax(y_true, 1)
    y_pred = np.argmax(y_pred, 1)

    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    print('\nAccuracy: ', accuracy_score(y_true, y_pred))
    print('Recall: ', recall_score(y_true, y_pred, average='weighted'))
    print('Precision: ', precision_score(y_true, y_pred, average='weighted'))
    print('F1-Score: ', f1_score(y_true, y_pred, average='weighted'))

    
def adversarial_attack(fool_model, x_test, y_test, max_num_attacks=100, epsilon=0.01, debug=False):

    # Run the attack and create and adversarial image
    print("Run attack for epsilon = " + str(epsilon))

    fmodel = foolbox.models.KerasModel(fool_model, bounds=(0, 1))
    num_attacks = 0
    num_success_attacks = 0
    for test_id in range(max_num_attacks):
        sys.stdout.write("\rRunning attack: {0}%".format(int(test_id * 100 / max_num_attacks)))
        sys.stdout.flush()

        x_true, y_true = x_test[test_id], np.argmax(y_test[test_id])
        
        # Run attack only if original prediciton was ok
        y_prediction = np.argmax(fool_model.predict(np.array([x_true])))
        if(y_prediction != y_true):
            continue

        # Run attack
        attack = foolbox.attacks.FGSM(fmodel)
        x_adversarial = attack(x_true, y_true, epsilons=[epsilon])
        num_attacks += 1

        # Check if an adversarial image was found
        if x_adversarial is None:
            continue
        
        # Convert into right format and get diff
        x_adversarial = x_adversarial[:, :, ::-1]  # convert BGR to RGB
        x_difference = x_adversarial - x_true
        x_difference = x_difference / abs(x_difference).max() * 0.2 + 0.5

        # Now lets predict using our model and measure some criteria
        y_adversarial = np.argmax(fool_model.predict(np.array([x_adversarial])))
        if(y_adversarial != y_true):
            num_success_attacks += 1

        # Show image of attack. But at most 1 image otherwise its too much for debugging...
        if debug:
            img = utils.stack_images([x_true, x_adversarial, x_difference], 3)
            img = img.resize((img.width*5, img.height*5), Image.ANTIALIAS)
            img.show()
            debug = False
    
    # Print results
    if num_attacks == 0:
        print("(Warning) No attack executed. Possible all predictions where wrong.")
    else:
        print("\n_______________________________________________")
        print("Num attacks: " + str(num_attacks))
        print("Num successfull attacks: " + str(num_success_attacks))
        print("Successrate [%]: " + str(num_success_attacks / num_attacks * 100))


#
# Main
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional Neural Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)

    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--max_num_samples', default=None, type=int,
                        help="Max. number of training examples to use. -1 to use all")

    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")

    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")

    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")

    parser.add_argument('--rotation_range', default=0.0, type=float,
                        help="(TestOnly) Rotate the test dataset randomly in the given range in degrees.")

    parser.add_argument('--crop_x', default=None, type=int,
                        help="Pixels to crop randomly into x direction.")

    parser.add_argument('--crop_y', default=None, type=int,
                        help="Pixels to crop randomly into x direction.")

    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")

    parser.add_argument('-f', '--fool', action='store_true',
                        help="Run adversarial attacks on the trained model. So provide weights via -w.")

    parser.add_argument('--save_dir', default='./result-convnet')

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    main(args)