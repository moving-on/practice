import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

mnist = input_data.read_data_sets('./mnist', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, timesteps, num_input])
y = tf.placeholder(tf.int32, shape=[None, num_classes])

weights = {'out':tf.Variable(tf.random_normal([2*num_hidden, num_classes]))}
biases = {'out':tf.Variable(tf.random_normal([num_classes]))}

def BiRNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bw_cell, fw_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights["out"]) + biases["out"]

output = BiRNN(x, weights, biases)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(y,1), predictions=tf.argmax(output, 1))[1]

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

for step in range(training_steps):
    b_x, b_y = mnist.train.next_batch(batch_size)
    b_x = b_x.reshape((batch_size, timesteps, num_input))
    _, l, acc = sess.run([train, loss, accuracy], {x:b_x, y:b_y})
    if step % display_step == 0:
        print "step:%d: loss=%f accuracy=%f" %(step, l, acc)

test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
