from __future__ import print_function
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import initializers
from tensorflow.keras.optimizers import Adam
from tensorflow.train import AdamOptimizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from core import BayesNN, average_gradients
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--samples", type=int, default=3)
parser.add_argument("-l", "--learningrate", type=float, default=0.001)
parser.add_argument("-a", "--activation", type=str, default="elu")
parser.add_argument("-c", "--batches", type=int, default=128)
parser.add_argument("-p", "--prior", nargs='+', default=[0, -6, 0.25])
parser.add_argument("-k", "--kernel", nargs='+', default=[-1, 1, -5, -4])
parser.add_argument("-b", "--bias", nargs='+', default=[-1, 1, -5, -4])
args = parser.parse_args()

print(args)

EPS = 1e-6

input_dim=(28,28)
output_dim=10
batch_size = args.batches
samples = int(args.samples)
learning_rate = args.learningrate
summaries_dir = './summaries/run-' + args.name + '-' + str(int(time.time()))
log_dir = './logs/run-' + args.name + '-' + str(int(time.time())) + '.txt'
with open(log_dir, 'w') as log_file:
    log_file.write(str(args) + '\n')
num_classes = 10
epochs = 50

model = BayesNN(input_dim,
                output_dim,
                batch_size=batch_size,
                activation = args.activation,
                prior_mixture_std_1 = np.exp(float(args.prior[0])).astype(np.float32),
                prior_mixture_std_2 = np.exp(float(args.prior[1])).astype(np.float32),
                prior_mixture_weight = float(args.prior[2]),
                kernel_mean_initializer=initializers.RandomUniform(minval=float(args.kernel[0]),
                                                                   maxval=float(args.kernel[1])),
                kernel_rho_initializer=initializers.RandomUniform(minval=float(args.kernel[2]),
                                                                  maxval=float(args.kernel[3])),
                bias_mean_initializer=initializers.RandomUniform(minval=float(args.bias[0]),
                                                                 maxval=float(args.bias[1])),
                bias_rho_initializer=initializers.RandomUniform(minval=float(args.bias[2]),
                                                                maxval=float(args.bias[3])))


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_mean = np.mean(x_train, axis = 0)
#x_std = np.std(x_train, axis = 0)
#x_train = (x_train.astype(np.float32) - x_mean) / (x_std + EPS)
#x_test = (x_test.astype(np.float32) - x_mean) / (x_std + EPS)
random_indices = np.random.permutation(len(x_train))[:500]
x_train = x_train[random_indices].astype(np.float32)
y_train = y_train[random_indices]
x_test = x_test.astype(np.float32)

N = len(x_train)
M = N // batch_size

# Convert class vectors to binary class matrices.
y_train_logits = keras.utils.to_categorical(y_train, num_classes)
y_test_logits = keras.utils.to_categorical(y_test, num_classes)

cce = CategoricalCrossentropy()
optimizer = AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay=0.00001, learning_rate=learning_rate)
acc = tf.keras.metrics.Accuracy()

in_ph = tf.placeholder(name='input', shape=(None, 28, 28), dtype=tf.float32)
labels_ph = tf.placeholder(name='labels', shape=(None), dtype=tf.int32)

sampled_predictions = []
sampled_complexity_losses = []
sampled_likelihood_losses = []
sampled_losses = []
sampled_grads = []

for i in range(50):
    predictions, complexity_loss = model(in_ph)
    sampled_predictions.append(predictions)

    if i < samples:
        likelihood_loss = tf.losses.sparse_softmax_cross_entropy(labels_ph, predictions)
        weight = 1/M
        loss = weight * complexity_loss + likelihood_loss
        grad = optimizer.compute_gradients(loss)

        sampled_complexity_losses.append(complexity_loss)
        sampled_likelihood_losses.append(likelihood_loss)
        sampled_losses.append(loss)
        sampled_grads.append(grad)

stacked_predictions = tf.stack(sampled_predictions)
stacked_predictions_train = tf.stack(sampled_predictions[:samples])
avg_prediction = tf.reduce_mean(tf.nn.softmax(stacked_predictions), axis=0)
avg_prediction_train = tf.reduce_mean(tf.nn.softmax(stacked_predictions_train), axis=0)

entropy_train = -tf.reduce_sum((avg_prediction_train * tf.log(avg_prediction_train+EPS)), axis=1)
entropy = -tf.reduce_sum((avg_prediction * tf.log(avg_prediction+EPS)), axis=1)
avg_entropy = tf.reduce_mean(entropy_train)
_, most_confusing = tf.math.top_k(entropy, k = 10)
_, most_confident = tf.math.top_k(-entropy, k = 10)

avg_complexity_loss = tf.reduce_mean(tf.stack(sampled_complexity_losses))
avg_likelihood_loss = tf.reduce_mean(tf.stack(sampled_likelihood_losses))
avg_loss = tf.reduce_mean(tf.stack(sampled_losses))
avg_grads = average_gradients(sampled_grads)


prediction = tf.argmax(avg_prediction, axis=1, output_type=tf.int32)
correct_prediction = tf.equal(prediction, y_test)
val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

step = tf.train.get_or_create_global_step()
update = optimizer.apply_gradients(avg_grads, global_step=step)

tf.summary.scalar('avg_complexity_loss', avg_complexity_loss)
tf.summary.scalar('avg_likelihood_loss', avg_likelihood_loss)
tf.summary.scalar('avg_entropy', avg_entropy)
tf.summary.scalar('avg_loss', avg_loss)

init = tf.global_variables_initializer()

summaries_op = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter(summaries_dir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        print("EPOCH {}\n".format(epoch))
        batches = np.random.permutation(range(N))[:N-(N % batch_size)].reshape(M, batch_size)
        for i in tqdm(range(len(batches))):
            batch = batches[i]
            feed_dict = {in_ph : x_train[batch], labels_ph : y_train[batch]}
            summaries, _ = sess.run([summaries_op, update], feed_dict)
            summary_writer.add_summary(summaries, global_step=i+epoch*len(batches))
        va, top_confusing, top_confident, avp = sess.run([val_acc, most_confusing, most_confident, avg_prediction], {in_ph: x_test, labels_ph: y_test})
        print("Validation accuracy: {}".format(va))
        with open(log_dir, 'a') as log_file:
            log_file.write("EPOCH {}\n".format(epoch))
            log_file.write("Validation accuracy: {}\n".format(va))
        # print("Most confused: {}".format(top_confusing))
        # print("Most confused vals: {}".format(avp[top_confusing]))
        # print("Most confident: {}".format(top_confident))
        # print("Most confident vals: {}".format(avp[top_confident]))
