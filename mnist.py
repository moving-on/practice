import tensorflow as tf
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./mnist", one_hot=True)

xs=tf.placeholder(dtype=tf.float32, shape=[None, 784])
ys=tf.placeholder(dtype=tf.float32, shape=[None, 10])


hidden=tf.layers.dense(xs, 200, tf.nn.relu)
output=tf.layers.dense(hidden, 10, tf.nn.softmax)

loss=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(output), reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
accuracy=tf.metrics.accuracy(labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1),)[1]

init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        x,y=mnist.train.next_batch(100)
        _, l, acc=sess.run([optimizer, loss, accuracy], feed_dict={xs:x, ys:y})
        print "step[%d]:loss=%f, acc=%f" %(step, l, acc)
