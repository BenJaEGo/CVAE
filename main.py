from tensorflow.examples.tutorials.mnist import input_data
from conditional_variational_auto_encoder import *
from vis_utils import *
import os
import numpy as np


def run_training():
    save_path = 'out'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    n_input = 784
    n_label = 10
    n_encoder_units = [100]
    n_decoder_units = [100]
    n_latent = 20
    lam = 0.0001
    lr = 0.001

    desired_label = 1

    max_epoch = 4000
    batch_size = 256
    n_sample, n_dims = mnist.train.images.shape
    n_batch_each_epoch = n_sample // batch_size

    graph = tf.Graph()

    with graph.as_default():

        model = ConditionalVariationalAutoEncoder(n_input, n_encoder_units, n_decoder_units, n_latent, n_label, lr, lam)

        with tf.Session(graph=graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(max_epoch):
                aver_loss = 0.0
                for step in range(n_batch_each_epoch):
                    x, y = mnist.train.next_batch(batch_size)

                    tr_loss, _ = sess.run(
                        fetches=[model.loss, model.train_op],
                        feed_dict={model.x_pl: x,
                                   model.y_pl: y}
                    )
                    aver_loss += tr_loss
                    # print("epoch %d, batch %d, tr_loss %f" % (epoch, step, tr_loss))
                print("epoch %d, loss %f" % (epoch, aver_loss / n_batch_each_epoch))

                n_sample = 16
                conditional_y = np.zeros([n_sample, n_label])
                conditional_y[:, desired_label] = 1
                latent = np.random.normal(loc=0.0, scale=1.0, size=[n_sample, n_latent])
                samples = sess.run(fetches=[model.reconstruct_x],
                                   feed_dict={model.latent_z: latent,
                                              model.y_pl: conditional_y})
                fig = visualize_generate_samples(samples[0])
                plt.savefig('{path}/epoch_{epoch}.png'.format(
                    path=save_path, epoch=epoch), bbox_inches='tight')
                plt.close(fig)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
