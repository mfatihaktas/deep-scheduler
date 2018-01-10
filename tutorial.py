import math, time, random
import numpy as np
import tensorflow as tf

def train1():
  feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]
  
  estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
  
  x_train = np.array([1., 2., 3., 4.])
  y_train = np.array([0., -1., -2., -3.])
  x_eval = np.array([2., 5., 8., 1.])
  y_eval = np.array([-1.01, -4.1, -7, 0.])
  input_fn = tf.estimator.inputs.numpy_input_fn(
      {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)
  
  estimator.train(input_fn=input_fn, steps=1000)
  
  train_metrics = estimator.evaluate(input_fn=train_input_fn)
  eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
  print("train metrics: %r"% train_metrics)
  print("eval metrics: %r"% eval_metrics)

def gradient():
  sess = tf.Session()
  
  x = tf.placeholder(tf.float32)
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  linear_model = W*x + b
  
  init = tf.global_variables_initializer()
  sess.run(init)
  
  y = tf.placeholder(tf.float32)
  loss = tf.reduce_sum(tf.square(linear_model - y) )
  
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss)
  for i in range(10):
    sess.run(train, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3] } )
    print(sess.run([W, b] ) )
    g = tf.gradients(linear_model, [W, b], grad_ys=None, name='gradients', colocate_gradients_with_ops=False, gate_gradients=False, aggregation_method=None)
    print(sess.run(g, feed_dict={x: [1], y: [1] } ) )
  
  # r = sess.run(loss, feed_dict={x: [1, 2, 3, 4], y: [0, -1, -2, -3] } )
  # print("r= {}".format(r) )

def neural_net():
  input_size = 10
  output_size = input_size
  input_ph = tf.placeholder(tf.float32, shape=(None, input_size) )
  desired_output_ph = tf.placeholder(tf.float32, shape=(None, input_size) )
  
  # Neural net
  hidden1_size, hidden2_size = 10, 10
  with tf.name_scope('hidden1'):
    w = tf.Variable(
          tf.truncated_normal([input_size, hidden1_size], stddev=1.0 / math.sqrt(float(input_size) ) ),
          name='weights')
    tf.summary.histogram('histogram', w)
    b = tf.Variable(tf.zeros([hidden1_size] ), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(input_ph, w) + b)
  with tf.name_scope('hidden2'):
    w = tf.Variable(
          tf.truncated_normal([hidden1_size, hidden2_size], stddev=1.0 / math.sqrt(float(hidden2_size) ) ),
          name='weights')
    tf.summary.histogram('histogram', w)
    b = tf.Variable(tf.zeros([hidden2_size] ), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
  with tf.name_scope('output'):
    w = tf.Variable(
        tf.truncated_normal([hidden2_size, output_size], stddev=1.0 / math.sqrt(float(hidden2_size) ) ),
        name='weights')
    tf.summary.histogram('histogram', w)
    b = tf.Variable(tf.zeros([output_size] ), name='biases')
    # output = tf.matmul(hidden2, w) + b
    output = tf.nn.softmax(tf.matmul(hidden2, w) + b)
  
  # loss = tf.norm(output - input_ph, ord=1) # tf.reduce_sum(tf.square(output - input_ph) )
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_ph, logits=output, name='xentropy') )
  loss = tf.reduce_sum(input_ph * -tf.log(output) )
  tf.summary.scalar('loss', loss)
  
  learning_rate = 0.01
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  
  # eval_correct = tf.nn.in_top_k(output, tf.argmax(input_ph), 1)
  eval_correct = tf.equal(tf.argmax(output, axis=1), tf.argmax(input_ph, axis=1) )
  
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  
  summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter("/home/ubuntu/deep-scheduler/log", sess.graph)
  
  def gen_in_data():
    in_data = np.zeros((1, input_size) )
    in_data[0, random.randint(0, input_size-1) ] = 1
    # in_data = np.zeros((100, input_size) )
    # for i in range(100):
    #   in_data[i, random.randint(0, input_size-1) ] = 1
    return in_data
  
  for step in range(10):
    start_time = time.time()
    
    feed_dict = {input_ph: gen_in_data() }
    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    duration = time.time() - start_time
    if step % 100 == 0:
      print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration) )
      # Update the events file.
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
  
  # Evaluate
  succ = 0
  for _ in range(5):
    i = gen_in_data()
    # print("i= {}".format(sess.run(tf.argmax(i, axis=1), feed_dict={input_ph: i} ) ) )
    # print("o= {}".format(sess.run(tf.argmax(output, axis=1), feed_dict={input_ph: i} ) ) )
    e = sess.run(eval_correct, feed_dict={input_ph: i} )
    # print("e= {}".format(e) )
    if e:
      succ += 1
    
    # hidden1_vs = sess.run(, feed_dict={input_ph: i} )
    hidden1_vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden1')
    hidden2_vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='hidden2')
    # for v in hidden1_vs:
    #   print("v= {}".format(v.name) )
    # print("hidden1_vs= {}".format(hidden1_vs) )
    
    # hidden1_gradient = sess.run(tf.gradients(loss, hidden1_vs),
    #                             feed_dict={input_ph: np.ones((1, input_size) ) } )
    # print("hidden1_gradient= {}".format(hidden1_gradient) )
    hidden2_gradient = sess.run(tf.gradients(output, hidden2_vs),
                                feed_dict={input_ph: np.ones((1, input_size) ) } )
    print("hidden2_gradient= {}".format(hidden2_gradient) )
    
  
  print("success rate= {}".format(succ/100) )

if __name__ == "__main__":
  # train1()
  # gradient()
  
  neural_net()
