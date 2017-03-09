import tensorflow as tf

layer_1_neuron_count = 200
layer_2_neuron_count = 100
layer_3_neuron_count = 60
layer_4_neuron_count = 30
layer_5_neuron_count = 10
vector_length = 784

# image dimention
x = tf.placeholder(tf.float32, [None, 28, 28, 1])

# correct values
y_ = tf.placeholder(tf.float32, [None, 10])

def layer_1():

    w = tf.Variable(tf.truncated_normal([vector_length, layer_1_neuron_count], stddev=0.1))
    b = tf.Variable(tf.zeros([layer_1_neuron_count]))

    return w

sess = tf.Session()
t = layer_1()
sess.run(tf.global_variables_initializer())
print(sess.run(t))
