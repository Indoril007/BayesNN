from __future__ import print_function
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from core import BayesNN
import os

from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
args = parser.parse_args()


SAMPLES = 1
batch_size = 128
num_classes = 10
epochs = 100
data_augmentation = False
bayesion = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

N = len(x_train)
M = N // batch_size

# Convert class vectors to binary class matrices.
y_train_logits = keras.utils.to_categorical(y_train, num_classes)
y_test_logits = keras.utils.to_categorical(y_test, num_classes)

model = BayesNN(input_dim=(28, 28), output_dim=10, batch_size=batch_size)
cce = CategoricalCrossentropy()
optimizer = Adam(learning_rate=0.003)
acc = tf.keras.metrics.Accuracy()

logdir = './summaries/run-' + args.name + '-' + str(int(time.time()))
summary_writer = tf.summary.create_file_writer(logdir)
epochs = 100
step = tf.Variable(0, name='step', trainable=False, dtype=tf.int64)

@tf.function
def train_step(images, labels, weight):
    with tf.GradientTape() as t:
        predictions, complexity_loss = model(images, samples=SAMPLES)
        # predictions = tf.reduce_mean(predictions, axis=0)
        # complexity_loss = tf.reduce_mean(complexity_losses, axis=0)
        #likelihood_losses = tf.TensorArray(tf.float32, predictions.shape[0], clear_after_read=False)
        #losses = tf.TensorArray(tf.float32, predictions.shape[0])
        #for i in tf.range(predictions.shape[0]):
        #    likelihood_losses = likelihood_losses.write(i, cce(labels, predictions[i]))
        #    losses = losses.write(i, weight * complexity_loss[i] + likelihood_losses.read(i))

        #likelihood_losses = likelihood_losses.stack()
        #losses = tf.reshape(losses.stack(), (-1,1))
        #loss = tf.reduce_mean(losses)
        likelihood_loss = cce(labels, predictions[0])
        loss = weight * complexity_loss[0] + likelihood_loss

    grads = t.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    tf.summary.scalar('likelihood_loss', tf.reduce_mean(likelihood_loss))
    tf.summary.scalar('complexity_loss', tf.reduce_mean(complexity_loss))
    return likelihood_loss, complexity_loss

#tf.summary.trace_on(graph=True, profiler=True)
#traced = False
with summary_writer.as_default():
    for epoch in range(epochs):
        print("EPOCH {}".format(epoch))
        batches = np.random.permutation(range(N))[:N-(N % batch_size)].reshape(M,batch_size)
        for i in tqdm(range(len(batches))):
            tf.summary.experimental.set_step(step)
            batch = batches[i]
            #weight = (2**(M - i + 1))/(2**M - 1)
            weight = 1 / M
            l_loss, c_loss = train_step(x_train[batch], y_train_logits[batch], weight)
            #if not traced:
            #    tf.summary.trace_export(
            #        name="train_step_traced",
            #        step=step,
            #        profiler_outdir=logdir)
            #    traced = True
            step.assign_add(1)

        logits, complexity_loss = model(x_test, samples=SAMPLES)
        logits = tf.reduce_mean(logits, axis=0)
        complexity_loss = tf.reduce_mean(complexity_loss, axis=0)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        val_acc = acc(prediction, y_test)
        print("Validation accuracy: {}".format(val_acc))

# model = Sequential()
#
# if bayesion:
#     model.add(Flatten())
#     model.add(Bayesion(800))
#     model.add(Activation('relu'))
#     model.add(Bayesion(800))
#     model.add(Activation('relu'))
#     model.add(Bayesion(num_classes))
#     model.add(Activation('softmax'))
# else:
#     model.add(Flatten())
#     model.add(Dense(800))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(800))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
# # initiate RMSprop optimizer
# opt = keras.optimizers.Adam()
#
# # Let's train the model using RMSprop
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
#
# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_test, y_test),
#               shuffle=True)
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         zca_epsilon=1e-06,  # epsilon for ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         # randomly shift images horizontally (fraction of total width)
#         width_shift_range=0.1,
#         # randomly shift images vertically (fraction of total height)
#         height_shift_range=0.1,
#         shear_range=0.,  # set range for random shear
#         zoom_range=0.,  # set range for random zoom
#         channel_shift_range=0.,  # set range for random channel shifts
#         # set mode for filling points outside the input boundaries
#         fill_mode='nearest',
#         cval=0.,  # value used for fill_mode = "constant"
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False,  # randomly flip images
#         # set rescaling factor (applied before any other transformation)
#         rescale=None,
#         # set function that will be applied on each input
#         preprocessing_function=None,
#         # image data format, either "channels_first" or "channels_last"
#         data_format=None,
#         # fraction of images reserved for validation (strictly between 0 and 1)
#         validation_split=0.0)
#
#     # Compute quantities required for feature-wise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(x_train)
#
#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(x_train, y_train,
#                                      batch_size=batch_size),
#                         epochs=epochs,
#                         validation_data=(x_test, y_test),
#                         workers=4)
#
# # Save model and weights
# if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)
#
## Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])