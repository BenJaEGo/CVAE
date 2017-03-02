from layers import *


class Encoder(object):
    def __init__(self, n_input, n_units, n_latent, n_label, activation):
        self._n_input = n_input
        self._n_units = n_units
        self._n_latent = n_latent
        self._n_label = n_label
        self._activation = activation
        self._weight_decay_loss = 0.0

        self._n_layer = len(n_units)
        self._hidden_layers = list()
        for layer_idx in range(self._n_layer):
            layer_name = "encoder_layer_" + str(layer_idx + 1)
            if layer_idx is 0:
                n_layer_input = n_input + n_label
            else:
                n_layer_input = n_units[layer_idx - 1]
            n_unit = n_units[layer_idx]
            self._hidden_layers.append(
                AffinePlusNonlinearLayer(layer_name, n_layer_input, n_unit, activation))

        layer_name = "encoder_mu_"
        self._encoder_mu = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_latent)
        layer_name = "encoder_sigma_"
        self._encoder_sigma = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_latent)

        for layer_idx in range(self._n_layer):
            self._weight_decay_loss += self._hidden_layers[layer_idx].get_weight_decay_loss()
        self._weight_decay_loss += self._encoder_mu.get_weight_decay_loss()
        self._weight_decay_loss += self._encoder_sigma.get_weight_decay_loss()

    def forward(self, input_tensor_x, input_tensor_y):
        # print(input_tensor_x)
        # print(input_tensor_y)
        input_tensor = tf.concat(values=[input_tensor_x, input_tensor_y], axis=1)
        # print(input_tensor)
        output_tensor = input_tensor
        for layer_idx in range(self._n_layer):
            output_tensor = self._hidden_layers[layer_idx].forward(output_tensor)
        mu = self._encoder_mu.forward(output_tensor)
        log_sigma_square = self._encoder_sigma.forward(output_tensor)

        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        log_sigma_square = 1e-6 + tf.nn.softplus(log_sigma_square)

        epsilon = tf.random_normal(shape=tf.shape(log_sigma_square),
                                   mean=0.0,
                                   stddev=1.0,
                                   name='epsilon')
        latent = tf.sqrt(tf.exp(log_sigma_square)) * epsilon + mu
        # latent = tf.exp(log_sigma_square / 2) * epsilon + mu

        # [batch_size x 1]
        kl_loss = -1 / 2 * tf.reduce_sum(1 + log_sigma_square - tf.square(mu) - tf.exp(log_sigma_square),
                                         reduction_indices=[1])

        return mu, log_sigma_square, latent, kl_loss

    @property
    def wd_loss(self):
        return self._weight_decay_loss
