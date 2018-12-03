import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

BATCH_SIZE =64
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 128

tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('./mnist', one_hot=True)
#plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
#plt.show()

with tf.variable_scope("Generator"):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])
    G_l1 = tf.layers.dense(G_in, 256, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, 28*28)

with tf.variable_scope("Descriminator"):
    real_in = tf.placeholder(tf.float32, [None, 28*28])/255.
    D_l0 = tf.layers.dense(real_in, 128, tf.nn.relu, name="l")
    prob0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name="out")

    D_l1 = tf.layers.dense(G_out, 128, tf.nn.sigmoid, name="l", reuse=True)
    prob1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name="out", reuse=True)

D_loss = -tf.reduce_mean(tf.log(prob0) - tf.log(prob1))
G_loss = tf.reduce_mean(tf.log(1-prob1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Descriminator"))
train_G = tf.train.AdamOptimizer(LR_G).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10000):
    x,y = mnist.train.next_batch(BATCH_SIZE)
    x = np.reshape(x, [-1, 28*28])
    noise = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_o, G_l, _ = sess.run([G_out, G_loss, train_G], {G_in:noise, real_in:x})

    if step % 10 == 0:
        D_l, _ = sess.run([D_loss, train_D], {G_in:noise, real_in:x})
        print "step[%d] g_loss:%.4f, d_loss:%.4f" %(step, G_l, D_l)

image = np.reshape(G_o[0], [28, 28])
plt.imshow(image, cmap='gray')
plt.show()
