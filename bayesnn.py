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
from core import BayesNN, average_gradients, DropoutBayesNN
import sys

np.set_printoptions(threshold=sys.maxsize)

from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-s", "--samples", type=int, default=1)
parser.add_argument("-S", "--savefrequency", type=int, default=25)
parser.add_argument("-l", "--learningrate", type=float, default=0.001)
parser.add_argument("-A", "--alpha", type=float, default=0.5)
parser.add_argument("-B", "--beta", type=float, default=0)
parser.add_argument("-a", "--activation", type=str, default="elu")
parser.add_argument("-c", "--batches", type=int, default=128)
parser.add_argument("-p", "--prior", nargs='+', default=[0, -6, 0, 0, 0.25])
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

# model = DropoutBayesNN(input_dim,
#                        output_dim,
#                        batch_size=batch_size,
#                        activation=args.activation)

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
# x_mean = np.mean(x_train, axis = 0)
# x_std = np.std(x_train, axis = 0)
# x_train = (x_train.astype(np.float32) - x_mean) / (x_std + EPS)
# x_test = (x_test.astype(np.float32) - x_mean) / (x_std + EPS)
#x_train = x_train.astype(np.float32)
random_indices = np.random.permutation(len(X_TRAIN))[:100]
x_train = X_TRAIN[random_indices].astype(np.float32)
y_train = Y_TRAIN[random_indices]
np.delete(X_TRAIN, random_indices, 0)
np.delete(Y_TRAIN, random_indices, 0)

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
pi_ph = tf.placeholder(name='alpha', shape=(), dtype=tf.float32)
alpha_ph = tf.placeholder(name='alpha', shape=(), dtype=tf.float32)
labels_ph = tf.placeholder(name='labels', shape=(None,), dtype=tf.int32)

labels_one_hot = tf.one_hot(labels_ph, 10)

sampled_predictions = []
sampled_complexity_losses = []
sampled_likelihood_losses = []
sampled_losses = []
sampled_grads = []

for i in range(30):
    with tf.variable_scope('sample-{}'.format(i)):
        predictions, complexity_loss = model(in_ph)
        sampled_predictions.append(predictions)
        likelihood_loss = tf.losses.softmax_cross_entropy(labels_one_hot, predictions)
        #likelihood_loss = tf.reduce_mean(-tf.reduce_sum(labels_one_hot * tf.log(tf.nn.softmax(predictions)+EPS), axis=1))
        #probs = tf.nn.softmax(predictions)
        #entropy = -tf.reduce_sum((probs * tf.log(probs+EPS)), axis=1)
        #entropy_loss = tf.reduce_mean(tf.reduce_sum((1-labels_one_hot) * probs, axis=1) * entropy)
        loss = alpha_ph * pi_ph * complexity_loss + (1 - alpha_ph) * likelihood_loss
        sampled_complexity_losses.append((1/M)*complexity_loss)
        sampled_likelihood_losses.append(likelihood_loss)
        sampled_losses.append(loss)

        if i < samples:
            grad = optimizer.compute_gradients(loss)
            sampled_grads.append(grad)

test_stacked_predictions_op = tf.stack(sampled_predictions)
train_stacked_predictions_op = tf.stack(sampled_predictions[:samples])
test_avg_prediction_op = tf.reduce_mean(tf.nn.softmax(test_stacked_predictions_op), axis=0)
train_avg_prediction_op = tf.reduce_mean(tf.nn.softmax(train_stacked_predictions_op), axis=0)

train_entropy_op = -tf.reduce_sum((train_avg_prediction_op * tf.log(train_avg_prediction_op+EPS)), axis=1)
test_entropy_op = -tf.reduce_sum((test_avg_prediction_op * tf.log(test_avg_prediction_op+EPS)), axis=1)

train_avg_entropy_op = tf.reduce_mean(train_entropy_op)
test_avg_entropy_op = tf.reduce_mean(test_entropy_op)

train_aleatoric_op = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(train_stacked_predictions_op) *
                                                   tf.log(tf.nn.softmax(train_stacked_predictions_op)+EPS),
                                                   axis=2), axis=0)
train_avg_aleatoric_op = tf.reduce_mean(train_aleatoric_op)

test_aleatoric_op = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(test_stacked_predictions_op) *
                                                  tf.log(tf.nn.softmax(test_stacked_predictions_op)+EPS),
                                                  axis=2), axis=0)

train_epistemic_op = train_entropy_op - train_aleatoric_op
train_avg_epistemic_op = tf.reduce_mean(train_epistemic_op)

test_epistemic_op = train_entropy_op - train_aleatoric_op
test_avg_epistemic_op = tf.reduce_mean(test_epistemic_op)

test_avg_aleatoric_op = tf.reduce_mean(test_aleatoric_op)

pre_entropy = tf.reduce_sum((1-labels_one_hot) * train_avg_prediction_op, axis=1) * train_entropy_op
entropy_loss = -tf.reduce_mean(pre_entropy)

_, most_confusing = tf.math.top_k(test_entropy_op, k = 10)
_, most_confident = tf.math.top_k(-test_entropy_op, k = 10)

train_avg_complexity_loss_op = tf.reduce_mean(tf.stack(sampled_complexity_losses[:samples]))
train_avg_likelihood_loss_op = tf.reduce_mean(tf.stack(sampled_likelihood_losses[:samples]))
train_avg_loss_op = tf.reduce_mean(tf.stack(sampled_losses[:samples]))
#train_avg_loss_op = tf.reduce_mean(tf.stack(sampled_losses[:samples])) + beta*entropy_loss

test_avg_complexity_loss_op = tf.reduce_mean(tf.stack(sampled_complexity_losses))
test_avg_likelihood_loss_op = tf.reduce_mean(tf.stack(sampled_likelihood_losses))
test_avg_loss_op = tf.reduce_mean(tf.stack(sampled_losses))
#test_avg_loss_op = tf.reduce_mean(tf.stack(sampled_losses))

avg_grads_op = average_gradients(sampled_grads)
#avg_grads_op = optimizer.compute_gradients(train_avg_loss_op)

test_prediction_op = tf.argmax(test_avg_prediction_op, axis=1, output_type=tf.int32)
test_correct_prediction_op = tf.equal(test_prediction_op, labels_ph)
test_avg_entropy_correct_op = tf.reduce_mean(tf.boolean_mask(test_entropy_op, test_correct_prediction_op))
test_std_entropy_correct_op = tf.math.reduce_std(tf.boolean_mask(test_entropy_op, test_correct_prediction_op))
test_avg_entropy_incorrect_op = tf.reduce_mean(tf.boolean_mask(test_entropy_op, tf.logical_not(test_correct_prediction_op)))
test_std_entropy_incorrect_op = tf.math.reduce_std(tf.boolean_mask(test_entropy_op, tf.logical_not(test_correct_prediction_op)))
test_avg_epistemic_incorrect_op = tf.reduce_mean(tf.boolean_mask(test_epistemic_op, tf.logical_not(test_correct_prediction_op)))
test_std_epistemic_incorrect_op = tf.math.reduce_std(tf.boolean_mask(test_epistemic_op, tf.logical_not(test_correct_prediction_op)))
test_avg_epistemic_correct_op = tf.reduce_mean(tf.boolean_mask(test_epistemic_op, test_correct_prediction_op))
test_std_epistemic_correct_op = tf.math.reduce_std(tf.boolean_mask(test_epistemic_op, test_correct_prediction_op))
test_avg_aleatoric_incorrect_op = tf.reduce_mean(tf.boolean_mask(test_aleatoric_op, tf.logical_not(test_correct_prediction_op)))
test_acc_op = tf.reduce_mean(tf.cast(test_correct_prediction_op, tf.float32))

update_op = optimizer.apply_gradients(avg_grads_op)

batch_summaries_op = tf.summary.merge([tf.summary.scalar('batch_avg_complexity_loss', train_avg_complexity_loss_op),
                                       tf.summary.scalar('batch_avg_likelihood_loss', train_avg_likelihood_loss_op),
                                       tf.summary.scalar('batch_avg_entropy_loss', entropy_loss),
                                       tf.summary.scalar('batch_avg_entropy', train_avg_entropy_op),
                                       tf.summary.scalar('batch_avg_aleatoric', train_avg_aleatoric_op),
                                       tf.summary.scalar('batch_alpha', alpha_ph),
                                       tf.summary.scalar('batch_avg_loss', train_avg_loss_op)])

epoch_summaries_op = tf.summary.merge([tf.summary.scalar('epoch_avg_complexity_loss', test_avg_complexity_loss_op),
                                       tf.summary.scalar('epoch_avg_likelihood_loss', test_avg_likelihood_loss_op),
                                       tf.summary.scalar('epoch_avg_entropy', test_avg_entropy_op),
                                       tf.summary.scalar('epoch_avg_loss', test_avg_loss_op),
                                       tf.summary.scalar('accuracy', test_acc_op),
                                       tf.summary.scalar('avg_entropy_correct', test_avg_entropy_correct_op),
                                       tf.summary.scalar('std_entropy_correct', test_std_entropy_correct_op),
                                       tf.summary.scalar('avg_aleatoric_incorrect', test_avg_aleatoric_incorrect_op),
                                       tf.summary.scalar('avg_epistemic_incorrect', test_avg_epistemic_incorrect_op),
                                       tf.summary.scalar('std_epistemic_incorrect', test_std_epistemic_incorrect_op),
                                       tf.summary.scalar('avg_epistemic_correct', test_avg_epistemic_correct_op),
                                       tf.summary.scalar('std_epistemic_correct', test_std_epistemic_correct_op),
                                       tf.summary.scalar('avg_entropy_incorrect', test_avg_entropy_incorrect_op),
                                       tf.summary.scalar('std_entropy_incorrect', test_std_entropy_incorrect_op)])

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
            feed_dict = {in_ph: x_train[batch], labels_ph : y_train[batch], pi_ph: pi, alpha_ph : alpha}
            batch_summaries, _ = sess.run([batch_summaries_op, update_op], feed_dict)
            batch_summary_writer.add_summary(batch_summaries, global_step=step)
            step += 1

        random_indices = np.random.permutation(len(x_train))[:10000]
        test_summaries, test_acc = sess.run([epoch_summaries_op, test_acc_op], feed_dict={in_ph: x_test,
                                                                                          labels_ph: y_test,
                                                                                          pi_ph: 1/M,
                                                                                          alpha_ph: alpha})
        train_summaries, train_acc = sess.run([epoch_summaries_op, test_acc_op],
                                              feed_dict={in_ph: x_train[random_indices],
                                                         labels_ph: y_train[random_indices],
                                                         pi_ph: 1/M,
                                                         alpha_ph: alpha})

        test_summary_writer.add_summary(test_summaries, global_step=epoch)
        train_summary_writer.add_summary(train_summaries, global_step=epoch)

        print("validation accuracy: {}".format(test_acc))
        print("training accuracy: {}".format(train_acc))

        if epoch > 300 and epoch % 100 == 0:
            r = np.random.permutation(len(X_TRAIN))[:5000]
            train_top_confusing = sess.run(most_confusing, {in_ph: X_TRAIN[r], labels_ph: Y_TRAIN[r]})
            x_train = np.concatenate([x_train, X_TRAIN[r[train_top_confusing]]])
            y_train = np.concatenate([y_train, Y_TRAIN[r[train_top_confusing]]])
            np.delete(X_TRAIN, r[train_top_confusing], 0)
            np.delete(Y_TRAIN, r[train_top_confusing], 0)
            N = len(x_train)
            M = N // batch_size

        # if (epoch > 0) and (epoch % save_frequency == 0):
        #     model.save_weights(save_dir + 'epoch-{}'.format(epoch))




