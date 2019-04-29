import tensorflow as tf
import time
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Flatten, Activation
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import activations
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.engine.input_spec import InputSpec

### This is a temporary patch and may possibly be removed in future versions of TF2
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


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
                 prior_mixture_std_1 = np.exp(-1).astype(np.float32),
                 prior_mixture_std_2 = np.exp(-6).astype(np.float32),
                 prior_mixture_weight = 0.25,
                 kernel_mean_initializer='glorot_uniform',
                 kernel_rho_initializer=initializers.RandomUniform(minval=-5, maxval=-4),
                 kernel_mean_regularizer=None,
                 kernel_rho_regularizer=None,
                 kernel_mean_constraint=None,
                 kernel_rho_constraint=None,
                 bias_mean_initializer='glorot_uniform',
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

        self.prior_mixture_std_1 = prior_mixture_std_1
        self.prior_mixture_std_2 = prior_mixture_std_2
        self.prior_mixture_weight = prior_mixture_weight

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

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

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

        else:
            self.bias = None

        # self.complexity_cost = self.add_variable(initializer='zeros',
        #                                           name='complexity_cost',
        #                                           trainable=False)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    # def call(self, inputs, N):
    #     input_dim, units  = self.kernel_mean.shape

    #     # epsilon should be resampled for each input
    #     kernel_epsilon = K.random_normal((N, input_dim, units))
    #     kernel_std = K.log(1+K.exp(self.kernel_rho))
    #     self.kernel = self.kernel_mean + kernel_std * kernel_epsilon
    #     self.log_posterior = K.sum(-0.5 * (K.pow(((self.kernel - K.stop_gradient(self.kernel_mean)) / K.stop_gradient(kernel_std)), 2) + K.log(2*np.pi) + 2*K.log(K.stop_gradient(kernel_std)))) # Gaussian likelihoods

    #     kernel_mix_1 = -0.5 * (K.pow((self.kernel / self.prior_mixture_std_1), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_1))
    #     kernel_mix_2 = -0.5 * (K.pow((self.kernel / self.prior_mixture_std_2), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_2))
    #     self.log_prior = K.sum(self.prior_mixture_weight * kernel_mix_1 + (1 - self.prior_mixture_weight) * kernel_mix_2)

    #     output = K.batch_dot(inputs[:, np.newaxis, :], self.kernel)
    #     output = K.squeeze(output, axis=1)

    #     if self.use_bias:
    #         bias_epsilon = K.random_normal((N, units))
    #         bias_std = K.log(1 + K.exp(self.bias_rho))
    #         self.bias = self.bias_mean + bias_std * bias_epsilon
    #         self.log_posterior += K.sum(-0.5 * (K.pow(((self.bias - K.stop_gradient(self.bias_mean)) / K.stop_gradient(bias_std)), 2) + K.log(2*np.pi) + 2*K.log(K.stop_gradient(bias_std)))) # Gaussian likelihoods

    #         bias_mix_1 = -0.5 * (K.pow((self.bias / self.prior_mixture_std_1), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_1))
    #         bias_mix_2 = -0.5 * (K.pow((self.bias / self.prior_mixture_std_2), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_2))
    #         self.log_prior += K.sum(self.prior_mixture_weight * bias_mix_1 + (1 - self.prior_mixture_weight) * bias_mix_2)
    #
    #        output = output + self.bias
    #    if self.activation is not None:
    #        output = self.activation(output)
    #
    #    #self.complexity_cost.assign(self.log_posterior - self.log_prior)
    #    self.add_loss(self.log_posterior - self.log_prior)
    #
    #    return output

    def call(self, inputs, N):
        input_dim, units  = self.kernel_mean.shape

        # epsilon should be resampled for each input
        kernel_epsilon = K.random_normal(self.kernel_mean.shape)
        kernel_std = K.log(1+K.exp(self.kernel_rho))
        self.kernel = self.kernel_mean + kernel_std * kernel_epsilon
        self.kernel_posterior = K.sum(-0.5 * (K.pow(((self.kernel - K.stop_gradient(self.kernel_mean)) / K.stop_gradient(kernel_std)), 2) + K.log(2*np.pi) + 2*K.log(K.stop_gradient(kernel_std)))) # Gaussian likelihoods

        kernel_mix_1 = -0.5 * (K.pow((self.kernel / self.prior_mixture_std_1), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_1))
        kernel_mix_2 = -0.5 * (K.pow((self.kernel / self.prior_mixture_std_2), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_2))
        self.log_prior = K.sum(K.log(self.prior_mixture_weight * K.exp(kernel_mix_1) + (1 - self.prior_mixture_weight) * K.exp(kernel_mix_2)))

        output = K.dot(inputs, self.kernel)

        if self.use_bias:
            bias_epsilon = K.random_normal((units,))
            bias_std = K.log(1 + K.exp(self.bias_rho))
            self.bias = self.bias_mean + bias_std * bias_epsilon
            self.bias_posterior = K.sum(-0.5 * (K.pow(((self.bias - K.stop_gradient(self.bias_mean)) / K.stop_gradient(bias_std)), 2) + K.log(2*np.pi) + 2*K.log(K.stop_gradient(bias_std)))) # Gaussian likelihoods
            self.log_posterior = self.kernel_posterior + self.bias_posterior

            bias_mix_1 = -0.5 * (K.pow((self.bias / self.prior_mixture_std_1), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_1))
            bias_mix_2 = -0.5 * (K.pow((self.bias / self.prior_mixture_std_2), 2) + K.log(2*np.pi) + 2*K.log(self.prior_mixture_std_2))
            self.log_prior += K.sum(K.log(self.prior_mixture_weight * K.exp(bias_mix_1) + (1 - self.prior_mixture_weight) * K.exp(bias_mix_2)))

            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)

        #self.complexity_cost.assign(self.log_posterior - self.log_prior)
        # print("log posterior : {}".format(self.log_posterior))
        # print("log prior : {}".format(self.log_prior))
        self.add_loss(self.log_posterior - self.log_prior)

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
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BayesNN(tf.keras.Model):

    def __init__(self, input_dim, output_dim, batch_size=32):
        super(BayesNN, self).__init__()
        self.flatten = Flatten(input_shape=input_dim)
        self.layer_1 = Bayesion(400)
        self.activation_1 = Activation('relu')
        self.layer_2 = Bayesion(400)
        self.activation_2 = Activation('relu')
        self.final_layer = Bayesion(output_dim)
        self.final_layer_activation = Activation('softmax')
        self.batch_size = batch_size


    @tf.function
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.layer_1(x, self.batch_size)
        x = self.activation_1(x)
        x = self.layer_2(x, self.batch_size)
        x = self.activation_2(x)
        x = self.final_layer(x, self.batch_size)
        return self.final_layer_activation(x), K.sum(self.losses)



if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    M = len(x_train) / 32
    model = BayesNN(input_dim=(28,28), output_dim=10)
    y_train = keras.utils.to_categorical(y_train, 10)
    cce = CategoricalCrossentropy()
    with tf.GradientTape() as t:
        predictions, categorical_loss = model(x_train[:128].astype(np.float32))
        likelihood_loss = cce(y_train[:32].astype(np.float32), predictions)
    grads = t.gradient(categorical_loss, model.trainable_variables)
    y_train = keras.utils.to_categorical(y_train, 10)
