import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

xx = tf.reshape(x, [-1, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# The model
y = tf.nn.softmax(tf.matmul(xx, w) + b)

# Correct values
y_ = tf.placeholder(tf.float32, [None, 10])

# Loss function -- Cross Entropy
cross_entropy = -tf.reduce_mean(y_ * tf.log(y)) * 1000.0
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Training
train = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# Tensor Session
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_data = {x: batch_xs, y_: batch_ys}
    session.run(train, feed_dict=train_data)

# Test Model
test_data = {x: mnist.test.images, y_: mnist.test.labels}
# Test function
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(str(round(session.run(accuracy, feed_dict=test_data)*100, 2))+"%")

session.close()
