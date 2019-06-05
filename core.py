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
        self.sampled_weights = []
        # epsilon should be resampled for each input
        self.kernel, self.kernel_epsilon = self.kernel_distribution.sample()
        self.variational_posterior = K.sum(self.kernel_distribution.log_likelihood(self.kernel))
        self.log_prior = K.sum(self.prior_distribution.log_prob(self.kernel))

        self.sampled_weights.append(self.kernel)

        output = K.dot(inputs, self.kernel)

        if self.use_bias:
            self.bias, self.bias_epsilon = self.bias_distribution.sample()
            self.variational_posterior += K.sum(self.bias_distribution.log_likelihood(self.bias))
            self.log_prior += K.sum(self.prior_distribution.log_prob(self.bias))

            self.sampled_weights.append(self.bias)

            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        self.add_loss(self.variational_posterior - self.log_prior)
        return output

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
        self.layer_1 = Bayesion(800,
                                prior_mixture_std_1=prior_mixture_std_1,
                                prior_mixture_std_2=prior_mixture_std_2,
                                prior_mixture_mu_1=prior_mixture_mu_1,
                                prior_mixture_mu_2=prior_mixture_mu_2,
                                prior_mixture_weight=prior_mixture_weight,
                                kernel_mean_initializer=kernel_mean_initializer,
                                kernel_rho_initializer=kernel_rho_initializer,
                                bias_mean_initializer=bias_mean_initializer,
                                bias_rho_initializer=bias_rho_initializer)

        self.activation_1 = Activation(activation)
        self.layer_2 = Bayesion(800,
                                prior_mixture_std_1=prior_mixture_std_1,
                                prior_mixture_std_2=prior_mixture_std_2,
                                prior_mixture_mu_1=prior_mixture_mu_1,
                                prior_mixture_mu_2=prior_mixture_mu_2,
                                prior_mixture_weight=prior_mixture_weight,
                                kernel_mean_initializer=kernel_mean_initializer,
                                kernel_rho_initializer=kernel_rho_initializer,
                                bias_mean_initializer=bias_mean_initializer,
                                bias_rho_initializer=bias_rho_initializer)

        self.activation_2 = Activation(activation)
        # self.layer_3 = Bayesion(1200,
        #                         prior_mixture_std_1=prior_mixture_std_1,
        #                         prior_mixture_std_2=prior_mixture_std_2,
        #                         prior_mixture_mu_1=prior_mixture_mu_1,
        #                         prior_mixture_mu_2=prior_mixture_mu_2,
        #                         prior_mixture_weight=prior_mixture_weight,
        #                         kernel_mean_initializer=kernel_mean_initializer,
        #                         kernel_rho_initializer=kernel_rho_initializer,
        #                         bias_mean_initializer=bias_mean_initializer,
        #                         bias_rho_initializer=bias_rho_initializer)
        #
        # self.activation_3 = Activation(activation)
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

        self.batch_size = batch_size
        self.output_dim = output_dim
        self.variational_posterior = None
        self.log_prior = None

        self.mus = []
        self.rhos = []
        self.built_ = False

    #@tf.function
    def call(self, inputs):


        sampled_weights = []
        epsilons = []

        x = self.flatten(inputs)

        ## LAYER 1
        x = self.layer_1(x)
        x = self.activation_1(x)
        sampled_weights += self.layer_1.sampled_weights
        epsilons.append(self.layer_1.kernel_epsilon)
        epsilons.append(self.layer_1.bias_epsilon)

        x = self.layer_2(x)
        x = self.activation_2(x)
        sampled_weights += self.layer_2.sampled_weights
        epsilons.append(self.layer_2.kernel_epsilon)
        epsilons.append(self.layer_2.bias_epsilon)

        #x = self.layer_3(x)
        #x = self.activation_3(x)

        x = self.final_layer(x)
        sampled_weights += self.final_layer.sampled_weights
        epsilons.append(self.final_layer.kernel_epsilon)
        epsilons.append(self.final_layer.bias_epsilon)

        variational_posterior = self.layer_1.variational_posterior + self.layer_2.variational_posterior + \
                                self.final_layer.variational_posterior

        log_prior = self.layer_1.log_prior + self.layer_2.log_prior + self.final_layer.log_prior

        if not self.built_:
            self.built_ = True
            self.mus.append(self.layer_1.kernel_mean)
            self.mus.append(self.layer_1.bias_mean)
            self.rhos.append(self.layer_1.kernel_rho)
            self.rhos.append(self.layer_1.bias_rho)

            self.mus.append(self.layer_2.kernel_mean)
            self.mus.append(self.layer_2.bias_mean)
            self.rhos.append(self.layer_2.kernel_rho)
            self.rhos.append(self.layer_2.bias_rho)

            self.mus.append(self.final_layer.kernel_mean)
            self.mus.append(self.final_layer.bias_mean)
            self.rhos.append(self.final_layer.kernel_rho)
            self.rhos.append(self.final_layer.bias_rho)

        return x, K.sum(self.losses), sampled_weights, epsilons, variational_posterior, log_prior


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
