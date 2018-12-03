import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate

# data
mnist = input_data.read_data_sets('./mnist', one_hot=True)              # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

tf_x = tf.placeholder(dtype=tf.float32, shape=[None, TIME_STEP*INPUT_SIZE])
image=tf.reshape(tf_x, [-1, INPUT_SIZE])
input=tf.layers.dense(image, 100, tf.nn.relu)
input=tf.reshape(input, [-1, TIME_STEP, 100])
tf_y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

rnn_cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (hc_x, h_n) = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=None, dtype=tf.float32, time_major=False)
output = tf.layers.dense(h_n, 10)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y, 1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for step in range(1200):
    x, y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x:x, tf_y:y})
    if step % 50 == 0:
        acc = sess.run(accuracy, {tf_x:test_x, tf_y:test_y})
        print "step[%d]: train loss: %.4f, accuracy: %.2f" %(step, loss_, acc)

test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')
