import tensorflow as tf
import time
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Flatten, Activation, Dropout, Dense
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.engine.input_spec import InputSpec

### This is a temporary patch and may possibly be removed in future versions of TF2
#from tensorflow.python.ops import control_flow_util
#control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

EPS = 1e-9

def get_summaries(sampled, labels_ph):
    stacked_predictions_op = tf.stack(sampled["prediction"])
    softmax_predictions_op = tf.nn.softmax(stacked_predictions_op)
    avg_prediction_op = tf.reduce_mean(softmax_predictions_op, axis=0)
    entropy_op = -tf.reduce_sum((avg_prediction_op * tf.log(avg_prediction_op+EPS)), axis=1)
    avg_entropy_op = tf.reduce_mean(entropy_op)
    max_entropy_op = tf.reduce_max(entropy_op)

    aleatoric_op = tf.reduce_mean(-tf.reduce_sum(softmax_predictions_op *
                                                      tf.log(softmax_predictions_op+EPS),
                                                      axis=2), axis=0)

    avg_variational_posterior = tf.reduce_mean(tf.stack(sampled["variational_posterior"]))
    avg_log_prior = tf.reduce_mean(tf.stack(sampled["log_prior"]))

    epistemic_op = entropy_op - aleatoric_op
    avg_epistemic_op = tf.reduce_mean(epistemic_op)
    max_epistemic_op = tf.reduce_max(epistemic_op)
    avg_aleatoric_op = tf.reduce_mean(aleatoric_op)

    _, top_entropy_indices = tf.math.top_k(entropy_op, k=1)
    _, top_epistemic_indices = tf.math.top_k(epistemic_op, k=1)

    avg_complexity_loss_op = tf.reduce_mean(tf.stack(sampled["complexity_loss"]))
    avg_likelihood_loss_op = tf.reduce_mean(tf.stack(sampled["likelihood_loss"]))
    avg_loss_op = tf.reduce_mean(tf.stack(sampled["loss"]))

    prediction_op = tf.argmax(avg_prediction_op, axis=1, output_type=tf.int32)
    correct_prediction_op = tf.equal(prediction_op, labels_ph)
    avg_entropy_correct_op = tf.reduce_mean(tf.boolean_mask(entropy_op, correct_prediction_op))
    std_entropy_correct_op = tf.math.reduce_std(tf.boolean_mask(entropy_op, correct_prediction_op))
    avg_entropy_incorrect_op = tf.reduce_mean(tf.boolean_mask(entropy_op, tf.logical_not(correct_prediction_op)))
    std_entropy_incorrect_op = tf.math.reduce_std(tf.boolean_mask(entropy_op, tf.logical_not(correct_prediction_op)))
    avg_epistemic_incorrect_op = tf.reduce_mean(tf.boolean_mask(epistemic_op, tf.logical_not(correct_prediction_op)))
    std_epistemic_incorrect_op = tf.math.reduce_std(tf.boolean_mask(epistemic_op, tf.logical_not(correct_prediction_op)))
    avg_epistemic_correct_op = tf.reduce_mean(tf.boolean_mask(epistemic_op, correct_prediction_op))
    std_epistemic_correct_op = tf.math.reduce_std(tf.boolean_mask(epistemic_op, correct_prediction_op))
    avg_aleatoric_incorrect_op = tf.reduce_mean(tf.boolean_mask(aleatoric_op, tf.logical_not(correct_prediction_op)))
    acc_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))

    confusion_matrix_op = tf.math.confusion_matrix(labels_ph, prediction_op)
    unique_op, _, count_op = tf.unique_with_counts(prediction_op)
    bias_op = count_op / tf.reduce_sum(count_op)


    batch_summaries_op = tf.summary.merge([tf.summary.scalar('batch_avg_complexity_loss', avg_complexity_loss_op),
                                           tf.summary.scalar('batch_avg_likelihood_loss', avg_likelihood_loss_op),
                                           tf.summary.scalar('batch_avg_variational_posterior', avg_variational_posterior),
                                           tf.summary.scalar('batch_avg_log_prior', avg_log_prior),
                                           tf.summary.scalar('batch_avg_entropy', avg_entropy_op),
                                           tf.summary.scalar('batch_max_entropy', max_entropy_op),
                                           tf.summary.scalar('batch_avg_aleatoric', avg_aleatoric_op),
                                           tf.summary.scalar('batch_avg_epistemic', avg_epistemic_op),
                                           tf.summary.scalar('batch_max_epistemic', max_epistemic_op),
                                           tf.summary.scalar('batch_avg_loss', avg_loss_op)])

    epoch_summaries_op = tf.summary.merge([tf.summary.scalar('epoch_avg_complexity_loss', avg_complexity_loss_op),
                                           tf.summary.scalar('epoch_avg_likelihood_loss', avg_likelihood_loss_op),
                                           tf.summary.scalar('epoch_avg_variational_posterior', avg_variational_posterior),
                                           tf.summary.scalar('epoch_avg_log_prior', avg_log_prior),
                                           tf.summary.scalar('epoch_avg_entropy', avg_entropy_op),
                                           tf.summary.scalar('epoch_max_entropy', max_entropy_op),
                                           tf.summary.scalar('epoch_avg_epistemic', avg_epistemic_op),
                                           tf.summary.scalar('epoch_max_epistemic', max_epistemic_op),
                                           tf.summary.scalar('epoch_avg_loss', avg_loss_op),
                                           tf.summary.scalar('accuracy', acc_op),
                                           tf.summary.scalar('avg_entropy_correct', avg_entropy_correct_op),
                                           tf.summary.scalar('std_entropy_correct', std_entropy_correct_op),
                                           tf.summary.scalar('avg_aleatoric_incorrect', avg_aleatoric_incorrect_op),
                                           tf.summary.scalar('avg_epistemic_incorrect', avg_epistemic_incorrect_op),
                                           tf.summary.scalar('std_epistemic_incorrect', std_epistemic_incorrect_op),
                                           tf.summary.scalar('avg_epistemic_correct', avg_epistemic_correct_op),
                                           tf.summary.scalar('std_epistemic_correct', std_epistemic_correct_op),
                                           tf.summary.scalar('avg_entropy_incorrect', avg_entropy_incorrect_op),
                                           tf.summary.scalar('std_entropy_incorrect', std_entropy_incorrect_op)])

    ops = {
        "acc_op": acc_op,
        "top_entropy_indices": top_entropy_indices,
        "top_epistemic_indices": top_epistemic_indices,
        "entropy_op": entropy_op,
        "epistemic_op": epistemic_op,
        "softmax_predictions_op": softmax_predictions_op,
        "confusion_matrix_op": confusion_matrix_op,
        "unique_op": unique_op,
        "bias_op": bias_op,
    }

    return batch_summaries_op, epoch_summaries_op, ops



def cross_entropy_loss(labels_ph, predictions):
    labels_one_hot = tf.one_hot(labels_ph, 10)
    pre_cross_entropy = labels_one_hot * tf.log(predictions+EPS)
    cross_entropy_loss = -tf.reduce_sum(pre_cross_entropy)
    return cross_entropy_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class Gaussian(object):

    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho

    @property
    def sigma(self):
        return K.log(1 + K.exp(self.rho))

    def sample(self):
        epsilon = K.random_normal(self.mu.shape)
        return K.stop_gradient(self.mu) + K.stop_gradient(self.sigma) * epsilon, epsilon

    def log_likelihood(self, x):
        return -0.5 * (K.pow(((x - self.mu) / self.sigma), 2)
                             + K.log(2*np.pi)
                             + 2*K.log(self.sigma))


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2, mu1=0, mu2=0):
        super().__init__()
        self.pi = pi
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho1 = K.log(K.exp(self.sigma1) - 1)
        self.rho2 = K.log(K.exp(self.sigma2) - 1)
        self.gaussian1 = Gaussian(self.mu1, self.rho1)
        self.gaussian2 = Gaussian(self.mu2, self.rho2)

    def log_prob(self, x):
        prob1 = K.exp(self.gaussian1.log_likelihood(x))
        prob2 = K.exp(self.gaussian2.log_likelihood(x))
        return K.log(self.pi * prob1 + (1 - self.pi) * prob2)

class Bayesion(Layer):
    """ A Bayesion Neural Network implemented with Bayes by Backprop
    [Weight Uncertaint in Neural Networks, C. Blundell 2015]

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 prior_mixture_std_1 = np.exp(0).astype(np.float32),
                 prior_mixture_std_2 = np.exp(-6).astype(np.float32),
                 prior_mixture_mu_1=0.0,
                 prior_mixture_mu_2=0.0,
                 prior_mixture_weight = 0.25,
                 kernel_mean_initializer=initializers.RandomUniform(minval=-1, maxval=1),
                 kernel_rho_initializer=initializers.RandomUniform(minval=-5, maxval=-4),
                 kernel_mean_regularizer=None,
                 kernel_rho_regularizer=None,
                 kernel_mean_constraint=None,
                 kernel_rho_constraint=None,
                 bias_mean_initializer=initializers.RandomUniform(minval=-1, maxval=1),
                 bias_rho_initializer=initializers.RandomUniform(minval=-5, maxval=-4),
                 bias_mean_regularizer=None,
                 bias_rho_regularizer=None,
                 bias_mean_constraint=None,
                 bias_rho_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Bayesion, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.prior_distribution = ScaleMixtureGaussian(prior_mixture_weight, prior_mixture_std_1, prior_mixture_std_2,
                                                       prior_mixture_mu_1, prior_mixture_mu_2)

        self.kernel_mean_initializer = kernel_mean_initializer
        self.kernel_rho_initializer = kernel_rho_initializer
        self.kernel_mean_regularizer= kernel_mean_regularizer
        self.kernel_rho_regularizer = kernel_rho_regularizer
        self.kernel_mean_constraint = kernel_mean_constraint
        self.kernel_rho_constraint  = kernel_rho_constraint
        self.bias_mean_initializer  = bias_mean_initializer
        self.bias_rho_initializer   = bias_rho_initializer
        self.bias_mean_regularizer  = bias_mean_regularizer
        self.bias_rho_regularizer   = bias_rho_regularizer
        self.bias_mean_constraint   = bias_mean_constraint
        self.bias_rho_constraint    = bias_rho_constraint

        self.sampled_weights = []
        self.kernel_epsilon = None
        self.bias_epsilon = None

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape.as_list()[-1]

        self.kernel_mean = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_mean_initializer,
                                      name='kernel_mean',
                                      regularizer=self.kernel_mean_regularizer,
                                      constraint=self.kernel_mean_constraint)

        # Standard Deviation of weights claculated as log(1 + exp(rho))
        self.kernel_rho = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_rho_initializer,
                                      name='kernel_rho',
                                      regularizer=self.kernel_rho_regularizer,
                                      constraint=self.kernel_rho_constraint)

        self.kernel_distribution = Gaussian(self.kernel_mean, self.kernel_rho)

        if self.use_bias:
            self.bias_mean = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_mean_initializer,
                                        name='bias_mean',
                                        regularizer=self.bias_mean_regularizer,
                                        constraint=self.bias_mean_constraint)

            self.bias_rho = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_rho_initializer,
                                        name='bias_rho',
                                        regularizer=self.bias_rho_regularizer,
                                        constraint=self.bias_rho_constraint)

            self.bias_distribution = Gaussian(self.bias_mean, self.bias_rho)

        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True


    def call(self, inputs):

        self.kernel, self.kernel_epsilon = self.kernel_distribution.sample()
        self.variational_posterior = K.sum(self.kernel_distribution.log_likelihood(self.kernel))
        self.log_prior = K.sum(self.prior_distribution.log_prob(self.kernel))

        output = K.dot(inputs, self.kernel)

        if self.use_bias:
            self.bias, self.bias_epsilon = self.bias_distribution.sample()
            self.variational_posterior += K.sum(self.bias_distribution.log_likelihood(self.bias))
            self.log_prior += K.sum(self.prior_distribution.log_prob(self.bias))

            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        return output

    def reinitialize_weights(self):
        session = K.get_session()
        self.kernel_mean.initializer.run(session=session)
        self.kernel_rho.initializer.run(session=session)
        self.bias_mean.initializer.run(session=session)
        self.bias_rho.initializer.run(session=session)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
        }
        base_config = super(Bayesion, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BayesNN(tf.keras.Model):

    def __init__(self,
                 input_dim,
                 output_dim,
                 layer_sizes=(800, 800),
                 batch_size=128,
                 activation = 'relu',
                 prior_mixture_std_1 = np.exp(0).astype(np.float32),
                 prior_mixture_std_2 = np.exp(-6).astype(np.float32),
                 prior_mixture_mu_1 = 0.0,
                 prior_mixture_mu_2 = 0.0,
                 prior_mixture_weight = 0.50,
                 kernel_mean_initializer=initializers.RandomUniform(minval=-1, maxval=1),
                 kernel_rho_initializer=initializers.RandomUniform(minval=-5, maxval=-4),
                 bias_mean_initializer=initializers.RandomUniform(minval=1, maxval=2),
                 bias_rho_initializer=initializers.RandomUniform(minval=-5, maxval=-4)):

        super(BayesNN, self).__init__()
        self.flatten = Flatten(input_shape=input_dim)

        self.model_layers = []
        self.model_activations = []

        for units in layer_sizes:
            self.model_layers.append(Bayesion(int(units),
                                     prior_mixture_std_1=prior_mixture_std_1,
                                     prior_mixture_std_2=prior_mixture_std_2,
                                     prior_mixture_mu_1=prior_mixture_mu_1,
                                     prior_mixture_mu_2=prior_mixture_mu_2,
                                     prior_mixture_weight=prior_mixture_weight,
                                     kernel_mean_initializer=kernel_mean_initializer,
                                     kernel_rho_initializer=kernel_rho_initializer,
                                     bias_mean_initializer=bias_mean_initializer,
                                     bias_rho_initializer=bias_rho_initializer))

            self.model_activations.append(Activation(activation))

        self.final_layer = Bayesion(output_dim,
                                prior_mixture_std_1=prior_mixture_std_1,
                                prior_mixture_std_2=prior_mixture_std_2,
                                prior_mixture_mu_1=prior_mixture_mu_1,
                                prior_mixture_mu_2=prior_mixture_mu_2,
                                prior_mixture_weight=prior_mixture_weight,
                                kernel_mean_initializer=kernel_mean_initializer,
                                kernel_rho_initializer=kernel_rho_initializer,
                                bias_mean_initializer=bias_mean_initializer,
                                bias_rho_initializer=bias_rho_initializer)

        self.variational_posterior = None
        self.log_prior = None
        self.complexity_loss = None
        self.likelihood_loss = None
        self.loss = None
        self.prediction = None

    def call(self, inputs):


        x = self.flatten(inputs)
        self.variational_posterior = 0
        self.log_prior = 0
        self.complexity_loss = 0
        self.likelihood_loss = None
        self.loss = None
        self.prediction = None

        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)
            x = self.model_activations[i](x)

            self.variational_posterior += self.model_layers[i].variational_posterior
            self.log_prior += self.model_layers[i].log_prior

        x = self.final_layer(x)
        self.variational_posterior += self.final_layer.variational_posterior
        self.log_prior += self.final_layer.log_prior
        self.complexity_loss = self.variational_posterior - self.log_prior

        self.prediction = x
        return x

    def grads(self, likelihood_loss, weight):
        loss = weight * self.complexity_loss + likelihood_loss
        grads = []

        for layer in self.model_layers + [self.final_layer]:
            layer_mus = [layer.kernel_mean, layer.bias_mean]
            layer_rhos = [layer.kernel_rho, layer.bias_rho]

            # Partial derivatives
            dl_dkw, dl_dbw = tf.gradients(loss, [layer.kernel, layer.bias]) # dLoss / dKernelWeight & dLoss / dBiasWeight
            dl_dkmu, dl_dbmu = tf.gradients(loss, layer_mus) # dLoss / dKernelMean & dLoss / dBiasMean
            dl_dkrho, dl_dbrho = tf.gradients(loss, layer_rhos) # dLoss / dKernelRho & dLoss / dBiasRho

            mu_grads = [dl_dkw + dl_dkmu, dl_dbw + dl_dbmu]
            rho_grads = [dl_dkw * (layer.kernel_epsilon / (1 + tf.exp(-layer.kernel_rho))) + dl_dkrho,
                         dl_dbw * (layer.bias_epsilon / (1 + tf.exp(-layer.bias_rho))) + dl_dbrho]

            grads += list(zip(mu_grads + rho_grads, layer_mus + layer_rhos))

        return grads

    def get_state(self):
        state = {
            "prediction": self.prediction,
            "complexity_loss": self.complexity_loss,
            "variational_posterior": self.variational_posterior,
            "log_prior": self.log_prior,
        }
        return state

    def reinitialize_weights(self):
        for layer in self.model_layers:
            layer.reinitialize_weights()
        self.final_layer.reinitialize_weights()




class DropoutBayesNN(tf.keras.Model):

    def __init__(self,
                 input_dim,
                 output_dim,
                 batch_size=128,
                 activation = 'relu',
                 dropout=0.5):

        super(DropoutBayesNN, self).__init__()
        self.flatten = Flatten(input_shape=input_dim)

        self.layer_1 = Dense(800)
        self.activation_1 = Activation(activation)
        self.dropout_1 = Dropout(dropout)

        self.layer_2 = Dense(800)
        self.activation_2 = Activation(activation)
        self.dropout_2 = Dropout(dropout)

        self.layer_3 = Dense(800)
        self.activation_3 = Activation(activation)
        self.dropout_3 = Dropout(dropout)

        self.final_layer = Dense(output_dim)

        self.batch_size = batch_size
        self.output_dim = output_dim

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.layer_1(x)
        x = self.activation_1(x)
        x = self.dropout_1(x, training=True)
        x = self.layer_2(x)
        x = self.activation_2(x)
        x = self.dropout_2(x, training=True)
        x = self.layer_3(x)
        x = self.activation_3(x)
        x = self.dropout_3(x, training=True)
        x = self.final_layer(x)
        return x, 0


if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    M = len(x_train) / 32
    model = BayesNN(input_dim=(28,28), output_dim=10)
    y_train = keras.utils.to_categorical(y_train, 10)
    #cce = CategoricalCrossentropy()
    #with tf.GradientTape() as t:
    #    predictions, categorical_loss = model(x_train[:128].astype(np.float32))
    #    likelihood_loss = cce(y_train[:32].astype(np.float32), predictions)
    #grads = t.gradient(categorical_loss, model.trainable_variables)
    #y_train = keras.utils.to_categorical(y_train, 10)
    inp = x_train[:32].astype(np.float32)
    model(inp)
