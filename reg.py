import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:, np.newaxis]
noise=np.random.normal(0, 0.1, size=x.shape)
y=np.power(x,2)+noise

tf_x=tf.placeholder(dtype=tf.float32, shape=x.shape)
tf_y=tf.placeholder(dtype=tf.float32, shape=y.shape)

hidden=tf.layers.dense(tf_x, 10, tf.nn.relu)
output=tf.layers.dense(hidden, 1)

loss=tf.losses.mean_squared_error(tf_y, output)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for step in range(100):
        _, l, pred=sess.run([optimizer, loss, output], feed_dict={tf_x:x, tf_y:y})
        print "step:%d: loss=%f" %(step, l)
