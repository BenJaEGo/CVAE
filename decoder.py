from layers import *


class Decoder(object):
    def __init__(self, n_latent, n_units, n_output, n_label, activation):
        self._n_latent = n_latent
        self._n_units = n_units
        self._n_output = n_output
        self._n_label = n_label
        self._activation = activation
        self._weight_decay_loss = 0.0

        self._n_layer = len(n_units)
        self._hidden_layers = list()
        for layer_idx in range(self._n_layer):
            layer_name = "decoder_layer_" + str(layer_idx + 1)
            if layer_idx is 0:
                n_layer_input = n_latent + n_label
            else:
                n_layer_input = n_units[layer_idx]
            n_unit = n_units[layer_idx]
            self._hidden_layers.append(
                AffinePlusNonlinearLayer(layer_name, n_layer_input, n_unit, activation))

        layer_name = "decoder_mu_"
        self._decoder_mu = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_output)
        layer_name = "decoder_sigma_"
        self._decoder_sigma = AffinePlusNonlinearLayer(layer_name, n_units[-1], n_output)

        for layer_idx in range(self._n_layer):
            self._weight_decay_loss += self._hidden_layers[layer_idx].get_weight_decay_loss()
        self._weight_decay_loss += self._decoder_mu.get_weight_decay_loss()
        self._weight_decay_loss += self._decoder_sigma.get_weight_decay_loss()

    def forward(self, input_tensor_y, input_tensor_latent):
        input_tensor = tf.concat_v2(values=[input_tensor_y, input_tensor_latent], axis=1)

        output_tensor = input_tensor
        for layer_idx in range(self._n_layer):
            output_tensor = self._hidden_layers[layer_idx].forward(output_tensor)
        # output_tensor = self._decoder_mu.forward(output_tensor)
        mu = self._decoder_mu.forward(output_tensor)
        log_sigma_square = self._decoder_sigma.forward(output_tensor)

        epsilon = tf.random_normal(shape=tf.shape(log_sigma_square),
                                   mean=0.0,
                                   stddev=1.0,
                                   name='epsilon')
        output_tensor = tf.sqrt(tf.exp(log_sigma_square)) * epsilon + mu
        # output_tensor = tf.exp(log_sigma_square / 2) * epsilon + mu

        return output_tensor

    @property
    def wd_loss(self):
        return self._weight_decay_loss
