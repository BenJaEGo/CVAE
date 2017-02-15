from encoder import *
from decoder import *


class ConditionalVariationalAutoEncoder(object):
    def __init__(self, n_input, n_encoder_units, n_decoder_units, n_latent, n_label, lr, lam):

        self._n_input = n_input
        self._n_encoder_units = n_encoder_units
        self._n_decoder_units = n_decoder_units
        self._n_latent = n_latent
        self._n_label = n_label

        self._x_pl = tf.placeholder(tf.float32, shape=[None, n_input], name='x_pl')
        self._y_pl = tf.placeholder(tf.float32, shape=[None, n_label], name='y_pl')
        self._enc = Encoder(n_input, n_encoder_units, n_latent, n_label, tf.nn.relu)
        self._dec = Decoder(n_latent, n_decoder_units, n_input, n_label, tf.nn.relu)

        self._mu, self._log_sigma_square, self._latent, enc_kl_loss = self._enc.forward(self._x_pl, self._y_pl)
        dec_output = self._dec.forward(self._y_pl, self._latent)
        self._reconstruct = tf.nn.sigmoid(dec_output)

        # [batch_size x 1]
        self._reconstruct_loss = -tf.reduce_sum(
            self._x_pl * tf.log(1e-10 + self._reconstruct) + (1 - self._x_pl) * tf.log(1e-10 + 1 - self._reconstruct),
            reduction_indices=[1])

        self._kl_loss = enc_kl_loss
        self._wd_loss = lam * (self._enc.wd_loss + self._dec.wd_loss)
        self._loss = tf.reduce_mean(self._kl_loss + self._reconstruct_loss, reduction_indices=[0]) + self._wd_loss
        global_step = tf.Variable(0, name="global_step", trainable=False)
        self._lr = tf.Variable(lr)
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(ys=self._loss, xs=trainable_variables)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, name="ADAM_optimizer")

        self._train_op = self._optimizer.apply_gradients(grads_and_vars=zip(grads, trainable_variables),
                                                         global_step=global_step,
                                                         name="train_op")

    @property
    def x_pl(self):
        return self._x_pl

    @property
    def y_pl(self):
        return self._y_pl

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def reconstruct_x(self):
        return self._reconstruct

    @property
    def latent_z(self):
        return self._latent

    @property
    def encoder_gaussian_mean(self):
        return self._mu
