from __future__ import print_function
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import initializers
from tensorflow.train import AdamOptimizer
from tensorflow.keras.losses import CategoricalCrossentropy
from core import BayesNN, average_gradients, get_summaries
import sys

np.set_printoptions(threshold=sys.maxsize)

#from tensorflow.python.ops import control_flow_util
#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--samples", type=int, default=1)
parser.add_argument("-i", "--initial_size", type=int, default=50)
parser.add_argument("-f", "--final_size", type=int, default=200)
parser.add_argument("-t", "--initial_iterations", type=int, default=1000)
parser.add_argument("-T", "--step_iterations", type=int, default=500)
parser.add_argument("-l", "--learningrate", type=float, default=0.0001)
parser.add_argument("-a", "--activation", type=str, default="elu")
parser.add_argument("-S", "--sampling_type", type=str, default="random", choices=["random", "entropy", "epistemic"])
parser.add_argument("-c", "--batches", type=int, default=128)
parser.add_argument("-p", "--prior", nargs='+', default=[-1, -7, 0, 0, 0.25])
parser.add_argument("-k", "--kernel", nargs='+', default=[-.1, .1, -5, -4])
parser.add_argument("-b", "--bias", nargs='+', default=[-.1, .1, -5, -4])
args = parser.parse_args()

print(args)

EPS = 1e-9

input_dim=(28,28)
output_dim=10
batch_size = args.batches
samples = int(args.samples)
learning_rate = args.learningrate
summaries_dir = './summaries/' + args.name + '-' + str(int(time.time())) + '/'
save_dir = './saves/' + args.name + '-' + str(int(time.time())) + '/'
log_dir = './logs/' + args.name + '-' + str(int(time.time())) + '.txt'
with open(log_dir, 'w') as log_file:
    log_file.write(str(args) + '\n')

num_classes = 10
epochs = 50000
alpha = args.alpha
beta = args.beta
save_frequency = args.savefrequency

model = BayesNN(input_dim,
                output_dim,
                batch_size=batch_size,
                activation = args.activation,
                prior_mixture_std_1 = np.exp(float(args.prior[0])).astype(np.float32),
                prior_mixture_std_2 = np.exp(float(args.prior[1])).astype(np.float32),
                prior_mixture_mu_1=float(args.prior[2]),
                prior_mixture_mu_2=float(args.prior[3]),
                prior_mixture_weight=float(args.prior[4]),
                kernel_mean_initializer=initializers.RandomUniform(minval=float(args.kernel[0]),
                                                                   maxval=float(args.kernel[1])),
                kernel_rho_initializer=initializers.RandomUniform(minval=float(args.kernel[2]),
                                                                  maxval=float(args.kernel[3])),
                bias_mean_initializer=initializers.RandomUniform(minval=float(args.bias[0]),
                                                                 maxval=float(args.bias[1])),
                bias_rho_initializer=initializers.RandomUniform(minval=float(args.bias[2]),
                                                                maxval=float(args.bias[3])))

# model.load_weights('/home/mil/james/workspace/synched/BayesNN/saves/testing_aleatoric_values_11_with_saves-1557570933/epoch-6')

# The data, split between train and test sets:
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(X_TRAIN, Y_TRAIN), (x_test, y_test) = mnist.load_data()
X_TRAIN = X_TRAIN.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
random_indices = np.random.permutation(len(X_TRAIN))[:args.initial_size]
x_train = X_TRAIN[random_indices].astype(np.float32)
y_train = Y_TRAIN[random_indices]
np.delete(X_TRAIN, random_indices, 0)
np.delete(Y_TRAIN, random_indices, 0)
#x_train = X_TRAIN
#y_train = Y_TRAIN

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
labels_ph = tf.placeholder(name='labels', shape=(None,), dtype=tf.int32)
complexity_weight_ph = tf.placeholder(name='complexity_weight', shape=(), dtype=tf.float32)

labels_one_hot = tf.one_hot(labels_ph, 10)

sampled = {
    "prediction": [],
    "complexity_loss": [],
    "likelihood_loss": [],
    "variational_posterior": [],
    "log_prior": [],
    "loss": [],
    "grad": [],
}

def cross_entropy_loss(labels_ph, predictions):
    labels_one_hot = tf.one_hot(labels_ph, 10)
    pre_cross_entropy = labels_one_hot * tf.log(predictions+EPS)
    cross_entropy_loss = -tf.reduce_sum(pre_cross_entropy)
    return cross_entropy_loss

for i in range(samples):
    with tf.variable_scope('sample-{}'.format(i)):
        predictions = model(in_ph)
        likelihood_loss = cross_entropy_loss(labels_ph, tf.nn.softmax(predictions))

        sampled["likelihood_loss"].append(likelihood_loss)
        sampled["loss"].append((1/M)*model.complexity_loss + likelihood_loss)
        sampled["grad"].append(model.grads(likelihood_loss, weight=1/M))

        for key, val in model.get_state().items():
            sampled[key].append(val)

avg_grads_op = average_gradients(sampled["grad"])
update_op = optimizer.apply_gradients(avg_grads_op)

batch_summaries_op, epoch_summaries_op, ops = get_summaries(sampled, labels_ph)

init = tf.global_variables_initializer()

batch_summary_writer = tf.summary.FileWriter(summaries_dir + 'batch', tf.get_default_graph())
train_summary_writer = tf.summary.FileWriter(summaries_dir + 'train')
test_summary_writer = tf.summary.FileWriter(summaries_dir + 'test')

step = 0

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        print("EPOCH {}\n".format(epoch))
        batches = np.random.permutation(range(N))[:N-(N % batch_size)].reshape(M, batch_size)
        for i in tqdm(range(len(batches))):
            #pi = (2**(M-(i+1))) / ((2**M) - 1)
            pi = 1/M
            batch = batches[i]
            feed_dict = {in_ph: x_train[batch], labels_ph : y_train[batch]}
            batch_summaries, _ = sess.run([batch_summaries_op, update_op], feed_dict)
            batch_summary_writer.add_summary(batch_summaries, global_step=step)
            step += 1

        if epoch > 1 and epoch % 250 == 0:
            random_indices = np.random.permutation(len(x_train))[:10000]

            test_summaries, test_acc = sess.run([epoch_summaries_op, ops["acc_op"]], feed_dict={in_ph: x_test,
                                                                                                labels_ph: y_test})
            train_summaries, train_acc = sess.run([epoch_summaries_op, ops["acc_op"]],
                                                  feed_dict={in_ph: x_train[random_indices],
                                                             labels_ph: y_train[random_indices]})

            test_summary_writer.add_summary(test_summaries, global_step=epoch)
            train_summary_writer.add_summary(train_summaries, global_step=epoch)

            print("validation accuracy: {}".format(test_acc))
            print("training accuracy: {}".format(train_acc))

        if epoch >= args.initial_iterations and epoch % args.step_iterations == 0 and len(x_train) < args.final_size:
            r = np.random.permutation(len(X_TRAIN))[:10000]

            if args.sampling_type == "random":
                train_top_confusing = np.arange(10)
            elif args.sampling_type == "entropy":
                train_top_confusing = sess.run(ops["top_entropy_indices"], {in_ph: X_TRAIN[r], labels_ph: Y_TRAIN[r]})
            elif args.sampling_type == "epistemic":
                train_top_confusing = sess.run(ops["top_epistemic_indices"], {in_ph: X_TRAIN[r], labels_ph: Y_TRAIN[r]})

            x_train = np.concatenate([x_train, X_TRAIN[r[train_top_confusing]]])
            y_train = np.concatenate([y_train, Y_TRAIN[r[train_top_confusing]]])
            np.delete(X_TRAIN, r[train_top_confusing], 0)
            np.delete(Y_TRAIN, r[train_top_confusing], 0)
            N = len(x_train)
            M = N // batch_size




