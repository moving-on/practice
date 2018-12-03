import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

def add_layer(input, in_size, out_size, activation=None):
    w=tf.Variable(tf.random_normal([in_size, out_size]))
    b=tf.Variable(tf.zeros([1, out_size])+0.1)
    output=tf.matmul(input, w)+b
    if activation is not None:
        output=activation(output)
    else:
        output=output
    return output

n_data=np.ones((100, 20))
x0=np.random.normal(n_data*2, 1)
y0=np.zeros(100)
x1=np.random.normal(n_data*-2, 1)
y1=np.ones(100)
x=np.vstack((x0, x1))
y=np.hstack((y0, y1))

tf_x=tf.placeholder(dtype=tf.float32, shape=x.shape)
tf_y=tf.placeholder(dtype=tf.int32, shape=y.shape)

hidden=add_layer(tf_x, 20, 10, tf.nn.relu)
output=add_layer(hidden, 10, 2)

loss=tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
accuracy=tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1),)[1]
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
#optimizer=tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9).minimize(loss)
#optimizer=tf.train.AdagradOptimizer(learning_rate=0.05).minimize(loss)
#optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)
#optimizer=tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(loss)
#optimizer=tf.train.AdamOptimizer().minimize(loss)
optimizer=tf.train.FtrlOptimizer(learning_rate=0.05).minimize(loss)

init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        _, l, acc, pred=sess.run([optimizer, loss, accuracy, output], feed_dict={tf_x:x, tf_y:y})
        print "step[%d]:loss=%f, acc=%f" %(step, l, acc)
