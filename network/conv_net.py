import numpy as np
import tensorflow as tf
import random

# Hyperparameters
input_dim = 21
input_channels = 5
number_amino_acids = 20
layer1_filter_size = 5
layer1_depth = 10
layer1_stride = 2
layer2_filter_size = 3
layer2_depth = 20
layer2_stride = 1
layer3_nodes = 10
output_dim = 1
learning_rate = 0.01
l2_regularization_constant = 0.01
batch_size = 10
iterations = 20000

# Setup TF Graph
x = tf.placeholder(tf.float32, [None, input_dim, input_dim, input_dim, input_channels])
x2 = tf.placeholder(tf.float32, [None, number_amino_acids])
y = tf.placeholder(tf.float32, [None, 1])

# Layer 1
W1 = tf.Variable(tf.truncated_normal([layer1_filter_size, layer1_filter_size, layer1_filter_size, input_channels, layer1_depth], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[layer1_depth]))
conv1 = tf.nn.conv3d(x, W1, [1, layer1_stride, layer1_stride, layer1_stride, 1], padding='SAME') + b1
layer1_out = tf.nn.relu(conv1)

# Layer 2
W2 = tf.Variable(tf.truncated_normal([layer2_filter_size, layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[layer2_depth]))
conv2 = tf.nn.conv3d(layer1_out, W2, [1, layer2_stride, layer2_stride, layer2_stride, 1], padding='SAME') + b2
layer2_out = tf.nn.relu(conv2)

# Layer 3
layer2_out_shape = layer2_out.get_shape().as_list()
layer2_out_dim = layer2_out_shape[1] * layer2_out_shape[2] * layer2_out_shape[3] * layer2_out_shape[4]
reshape_layer2_out = tf.reshape(layer2_out, [-1, layer2_out_dim])
ff_input = tf.concat(1, [reshape_layer2_out, x2])
W3 = tf.Variable(tf.truncated_normal([layer2_out_dim + number_amino_acids, layer3_nodes], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[layer3_nodes]))
layer3_aggregate = tf.matmul(ff_input, W3) + b3
layer3_out = tf.nn.relu(layer3_aggregate)

# Layer 4 (Output)
W4 = tf.Variable(tf.truncated_normal([layer3_nodes, output_dim], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[output_dim]))
network_output = tf.matmul(layer3_out, W4) + b4

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(network_output, y))
number_correct = tf.reduce_sum(tf.cast(tf.equal(tf.round(tf.sigmoid(network_output)), y), tf.float32))
number_increase_correct = tf.reduce_sum(tf.cast(tf.equal(2*tf.round(tf.sigmoid(network_output)), y + 1), tf.float32))
loss_with_l2 = loss + l2_regularization_constant * (tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4))
optimize = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_with_l2)
#optimize = tf.train.AdamOptimizer().minimize(loss_with_l2)

def train(training_x, training_x2, training_y, valid_x, valid_x2, valid_y):
  print "here"
  out_file = open("output.txt", 'w')
  percent_increase_train = float(np.count_nonzero(training_y)) / training_y.shape[0]
  percent_increase_valid = float(np.count_nonzero(valid_y)) / valid_y.shape[0]
  fprint("Percent Increase in Training Set: " + str(percent_increase_train), out_file)
  fprint("Percent Increase in Validation Set: " + str(percent_increase_valid), out_file)
  sess = tf.Session()
  init = tf.initialize_all_variables()
  sess.run(init)
  checkpoint = 0
  for i in xrange(iterations):
    batch_indices = random.sample(xrange(0, training_x.shape[0]), batch_size)
    batch_x = training_x[batch_indices, :, :, :, :]
    batch_x2 = training_x2[batch_indices, :]
    batch_y = training_y[batch_indices, :]
    sess.run(optimize, feed_dict={x: batch_x, x2: batch_x2, y: batch_y})
    print i
    if i % 10 == 0:
      loss_valid = sess.run(loss_with_l2, feed_dict={x: valid_x, x2: valid_x2, y: valid_y})
      correct_valid = sess.run(number_correct, feed_dict={x: valid_x, x2: valid_x2, y: valid_y})
      increase_correct_valid = sess.run(number_increase_correct, feed_dict={x: valid_x, x2: valid_x2, y: valid_y})
      loss_train = sess.run(loss_with_l2, feed_dict={x: training_x, x2: training_x2, y: training_y})
      correct_train = sess.run(number_correct, feed_dict={x: training_x, x2: training_x2, y: training_y})
      increase_correct_train = sess.run(number_increase_correct, feed_dict={x: training_x, x2: training_x2, y: training_y})

      accuracy_valid = correct_valid / valid_x.shape[0]
      accuracy_train = correct_train / training_x.shape[0]
      increase_accuracy_valid = increase_correct_valid / np.count_nonzero(valid_y)
      increase_accuracy_train = increase_correct_train / np.count_nonzero(training_y)
      decrease_accuracy_valid = (correct_valid - increase_correct_valid) / (valid_y.shape[0] - np.count_nonzero(valid_y))
      decrease_accuracy_train = (correct_train - increase_correct_train) / (training_y.shape[0] - np.count_nonzero(training_y))

      
      fprint("Iteration " +  str(i) + " Training Loss: " + str(loss_train), out_file)
      fprint("Iteration " +  str(i) + " Training Accuracy: " + str(accuracy_train), out_file)
      fprint("Iteration " + str(i) + " Training Increase Accuracy: " + str(increase_accuracy_train), out_file)
      fprint("Iteration " + str(i) + " Training Decrease Accuracy: " + str(decrease_accuracy_train), out_file)

      fprint("Iteration " + str(i) + " Validation Loss: " + str(loss_valid), out_file)
      fprint("Iteration " + str(i) + " Validation Accuracy: " + str(accuracy_valid), out_file)
      fprint("Iteration " + str(i) + " Validation Increase Accuracy: " + str(increase_accuracy_valid), out_file)
      fprint("Iteration " + str(i) + " Validation Decrease Accuracy: " + str(decrease_accuracy_valid), out_file)

      saver = tf.train.Saver()
      saver.save(sess, '../models/model.ckpt')
      if i > 0 and i % 100 == 0:
        checkpoint += 100
        saver.save(sess, '../models/model-' + str(checkpoint) + '.ckpt')

def fprint(string, f):
  f.write(string + '\n')
  print string

def print_training_indices(filename, training_indices):
  f_out = open(filename, 'w')
  for index in training_indices:
    f_out.write(str(index) + '\n')
  f_out.close()

if __name__ == '__main__':
  X_lattices = np.load('../data/datasets/resampled_multichannel_X_lattices.npy')
  X_mutations = np.load('../data/datasets/resampled_X_mutations.npy')
  Y_class = np.load('../data/datasets/resampled_Y_class.npy')
  #X_lattices = X_lattices[:, :, :, :, np.newaxis]
  n = X_lattices.shape[0]
  training_size = int(n * 0.7)
  training_indices = random.sample(xrange(0, n), training_size)
  print_training_indices('training_indices.txt', training_indices)
  training_X = X_lattices[training_indices, :, :, :, :]
  training_X2 = X_mutations[training_indices, :]
  training_Y = Y_class[training_indices, :]
  valid_X = np.delete(X_lattices, training_indices, 0)
  valid_X2 = np.delete(X_mutations, training_indices, 0)
  valid_Y = np.delete(Y_class, training_indices, 0)
  train(training_X, training_X2, training_Y, valid_X, valid_X2, valid_Y)
