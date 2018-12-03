import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE=50
LR=0.001

mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x=mnist.test.images[:2000]
test_y=mnist.test.labels[:2000]

print mnist.train.images.shape
print mnist.train.labels.shape
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

tf_x=tf.placeholder(dtype=tf.float32, shape=[None, 28*28])/255.
image=tf.reshape(tf_x, [-1,28,28,1])
tf_y=tf.placeholder(dtype=tf.int32, shape=[None, 10])

conv1=tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
pool1=tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

conv2=tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=5, strides=1, padding="same", activation=tf.nn.relu)
pool2=tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

flat=tf.reshape(pool2, [-1, 7*7*32])
output=tf.layers.dense(flat, 10)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op=tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))

from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    for step in range(600):
        x, y=mnist.train.next_batch(BATCH_SIZE)
        _, loss_ = sess.run([train_op, loss], {tf_x:x, tf_y:y})
        if step % 50 == 0:
            acc, flat_ = sess.run([accuracy, flat], {tf_x:test_x, tf_y:test_y})
            print "step:%d: loss=%f" %(step, loss_)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
                low_dim_embs = tsne.fit_transform(flat_[:plot_only, :])
                labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

test_output=tf.Session().run(output, {tf_x:test_x[:10]})
pred_y=tf.argmax(test_output, 1)
print pred_y
print np.argmax(test_y[:10], 1)

