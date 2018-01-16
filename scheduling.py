import math, time, random
import numpy as np
import tensorflow as tf

class ValueEster(object):
  def __init__(self, s_len, nn_len=10):
    self.s_len = s_len
    self.nn_len = nn_len
    
    self.gamma = 0.99
    self.init()
  
  def __repr__(self):
    return "ValueEster[s_len= {}]".format(self.s_len)
  
  def init(self):
    # T x s_len
    self.s_ph = tf.placeholder(shape=(None, self.s_len), dtype=tf.float32)
    # self.hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
    # self.hidden2 = tf.contrib.layers.fully_connected(self.hidden1, self.nn_len, activation_fn=tf.nn.relu)
    # self.v = tf.contrib.layers.fully_connected(self.hidden2, 1, activation_fn=tf.nn.relu)
    with tf.name_scope('hidden1'):
      w = tf.Variable(
            tf.truncated_normal([self.s_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.s_len) ) ),
            name='weights')
      b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      hidden1 = tf.nn.relu(tf.matmul(self.s_ph, w) + b)
    with tf.name_scope('hidden2'):
      w = tf.Variable(
            tf.truncated_normal([self.nn_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
            name='weights')
      b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
    with tf.name_scope('v_layer'):
      w = tf.Variable(
          tf.truncated_normal([self.nn_len, 1], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
          name='weights')
      b = tf.Variable(tf.zeros([1] ), name='biases')
      self.v = tf.matmul(hidden2, w) + b
    self.sampled_v = tf.placeholder(shape=(None, 1), dtype=tf.float32)
    # self.loss = tf.reduce_sum(tf.squared_difference(self.v, self.sampled_v) )
    self.loss = tf.losses.mean_squared_error(self.v, self.sampled_v)
    
    # self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    self.optimizer = tf.train.AdamOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def train_w_single_traj(self, t_s_l, t_r_l):
    # r_l keeps one step reward r's not v's
    # print("t_s_l= {}".format(t_s_l) )
    v_p1_l = self.sess.run(self.v,
                           feed_dict={self.s_ph: t_s_l[1:] } )
    # print("v_p1_l= {}".format(v_p1_l) )
    # print("t_r_l= {}".format(t_r_l) )
    v_l = np.add(t_r_l[:-1], np.multiply(self.gamma, v_p1_l) )
    # v_l = t_r_l[:-1]
    
    # print("v_l= {}".format(v_l) )
    _, loss = self.sess.run([self.train_op, self.loss],
                            feed_dict={self.s_ph: t_s_l[:-1],
                                       self.sampled_v: v_l} )
    print("ValueEster:: loss= {}".format(loss) )
  
  def get_v(self, t_s_l):
    return self.sess.run(self.v,
                         feed_dict={self.s_ph: t_s_l} )

class DeepScher(object):
  def __init__(self, s_len, a_len, nn_len=10, straj_training=False):
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    self.straj_training = straj_training
    
    self.v_ester = ValueEster(s_len)
    self.gamma = 0.99
    self.init()
  
  def __repr__(self):
    return "DeepScher[s_len= {}, a_len= {}]".format(self.s_len, self.a_len)
  
  def init(self):
    if self.straj_training:
      self.s_ph = tf.placeholder(tf.float32, shape=(None, self.s_len), name="s_ph")
      # with tf.name_scope('hidden1'):
      #   w = tf.Variable(
      #         tf.truncated_normal([1, self.s_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.s_len) ) ),
      #         name='weights')
      #   b = tf.Variable(tf.zeros([1, self.nn_len] ), name='biases')
      #   hidden1 = tf.nn.relu(tf.matmul(self.s_ph, w) + b)
      # with tf.name_scope('hidden2'):
      #   w = tf.Variable(
      #         tf.truncated_normal([1, self.nn_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
      #         name='weights')
      #   b = tf.Variable(tf.zeros([1, self.nn_len] ), name='biases')
      #   hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
      # with tf.name_scope('a_probs'):
      #   w = tf.Variable(
      #       tf.truncated_normal([1, self.nn_len, self.a_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
      #       name='weights')
      #   b = tf.Variable(tf.zeros([1, self.a_len] ), name='biases')
      #   self.a_probs = tf.nn.softmax(tf.matmul(hidden2, w) + b)
      hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
      self.a_probs = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=tf.nn.softmax)
      
      self.a_ph = tf.placeholder(tf.int32, shape=(None, 1), name="a_ph")
      self.q_ph = tf.placeholder(tf.float32, shape=(None, 1), name="q_ph")
      self.v_ph = tf.placeholder(tf.float32, shape=(None, 1), name="v_ph")
      
      sh = tf.shape(self.a_probs)
      indices = tf.range(0, sh[0] )*sh[1] + tf.reshape(self.a_ph, [-1] )
      self.resp_outputs = tf.gather(tf.reshape(self.a_probs, [-1]), indices)
      self.loss = -tf.reduce_mean(tf.log(self.resp_outputs)*(self.q_ph - self.v_ph) )
    else:
      # N x T x s_len
      self.s_ph = tf.placeholder(tf.float32, shape=(None, None, self.s_len) )
      hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
      self.a_probs = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=tf.nn.softmax)
      
      self.a_ph = tf.placeholder(tf.int32, shape=(None, None), name="a_ph")
      self.q_ph = tf.placeholder(tf.float32, shape=(None, None), name="q_ph")
      self.v_ph = tf.placeholder(tf.float32, shape=(None, None), name="v_ph")
      
      sh = tf.shape(self.a_probs)
      N, T = sh[0], sh[1]
      indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.a_ph, [-1] )
      self.resp_outputs = tf.reshape(tf.gather(tf.reshape(self.a_probs, [-1] ), indices), (sh[0], sh[1] ) )
      self.loss = -tf.reduce_mean(tf.log(self.resp_outputs)*(self.q_ph - self.v_ph) )
      # self.loss = -tf.reduce_sum(tf.reduce_mean(tf.log(self.resp_outputs)*(self.q_ph - self.v_ph), axis=0) )
    
    # self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    self.optimizer = tf.train.AdamOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def get_random_action(self, s):
    # print("s= {}".format(s) )
    if self.straj_training:
      a_probs = self.sess.run(self.a_probs,
                              feed_dict={self.s_ph: [s] } )
      a_dist = np.array(a_probs[0] )
    else:
      a_probs = self.sess.run(self.a_probs,
                              feed_dict={self.s_ph: [[s]] } )
      a_dist = np.array(a_probs[0][0] )
    # print("a_dist= {}".format(a_dist) )
    a = np.random.choice(a_dist, 1, p=a_dist)
    a = np.argmax(a_dist == a)
    # print("a= {}".format(a) )
    return a
  
  def get_max_action(self, s):
    if self.straj_training:
      a_probs = self.sess.run(self.a_probs, feed_dict={self.s_ph: [s] } )
      a_dist = a_probs[0]
    else:
      a_probs = self.sess.run(self.a_probs, feed_dict={self.s_ph: [[s]] } )
      a_dist = a_probs[0][0]
    # print("a_dist= {}".format(a_dist) )
    return np.argmax(a_dist)
  
  def discount_rewards(self, r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r) ) ):
      running_add = r[t] + running_add * self.gamma
      discounted_r[t] = running_add
    return discounted_r
  
  def train_w_single_traj(self, t_s_l, t_a_l, t_r_l):
    # print("t_r_l= {}".format(t_r_l) )
    # t_q_l = self.discount_rewards(t_r_l)
    # v = sum(t_q_l)/len(t_q_l)
    # T = len(t_r_l)
    # t_v_l = np.array([v]*T).reshape(T, 1)
    # t_v_l = [v]*len(t_r_l)
    # loss, _ = self.sess.run([self.loss, self.train_op],
    #                         feed_dict={self.s_ph: t_s_l,
    #                                   self.a_ph: t_a_l,
    #                                   self.q_ph: t_q_l,
    #                                   self.v_ph: t_v_l} )
    # T = len(t_a_l)
    # t_a_l = np.array(t_a_l).reshape(T, 1)
    
    # t_r_l = np.array(t_r_l).reshape(T, 1)
    # print("t_r_l= {}".format(t_r_l) )
    self.v_ester.train_w_single_traj(t_s_l, t_r_l)
    t_v_l = self.v_ester.get_v(t_s_l)
    t_vp1_l = t_v_l[1:]
    t_q_l = np.add(t_r_l[:-1], np.multiply(self.gamma, t_vp1_l) )
    # print("t_vp1_l= {}".format(t_vp1_l) )
    # print("t_v_l= {}".format(t_v_l) )
    # print("t_r_l= {}".format(t_r_l) )
    # t_v_l = np.reshape(t_v_l[:-1], (-1, 1) )
    # t_v_l = np.expand_dims(t_v_l[:-1], axis=1) # tf.reshape(t_v_l[:-1], (T-1, 1) )
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: t_s_l[:-1],
                                       self.a_ph: t_a_l[:-1],
                                       self.q_ph: t_q_l,
                                       self.v_ph: t_v_l[:-1] } )
    # print("sh= {}".format(sh) )
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    # All trajectories use the same policy
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    for n in range(N):
      n_t_r_l[n] = self.discount_rewards(n_t_r_l[n] )
    t_avgr_l = [np.mean(n_t_r_l[:, t] ) for t in range(T) ]
    n_t_b_l = np.array([t_avgr_l for _ in range(N) ] )
    # n_t_b_l = np.array([np.mean(n_t_r_l)] * N*T).reshape((N, T) )
    
    # print("n_t_a_l= {}".format(n_t_a_l) )
    # print("n_t_r_l= {}".format(n_t_r_l) )
    # print("n_t_b_l= {}".format(n_t_b_l) )
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                      self.a_ph: n_t_a_l,
                                      self.q_ph: n_t_r_l,
                                      self.v_ph: n_t_b_l} )
    # print("sh= {}".format(sh) )
    # print(">> loss= {}".format(loss) )
  
def test():
  s_len, a_len, nn_len = 3, 3, 10
  straj_training = True # False
  scher = DeepScher(s_len, a_len, nn_len, straj_training)
  
  def state():
    s = np.random.randint(10, size=s_len)
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def reward(s, a):
    # s_min = min(s)
    # r = 10 if s[a] == s_min else 0
    # return min(100, 1/(0.001 + s[a] - min(s) ) )
    # return 100*math.exp(-(s[a] - min(s) ) )
    return 1/(0.1 + s[a] - min(s) )
  
  def evaluate():
    num_shortest_found = 0
    for e in range(100):
      s = state()
      a = scher.get_max_action(s)
      if s[a] - min(s) < 0.01:
        num_shortest_found += 1
    print("freq shortest found= {}".format(num_shortest_found/100) )
  
  def train_w_single_traj():
    T = 100
    def gen_traj():
      t_s_l, t_a_l, t_r_l = np.zeros((T, s_len)), np.zeros((T, 1)), np.zeros((T, 1))
      for t in range(T):
        s = state()
        a = scher.get_random_action(s)
        # a = scher.get_max_action(s)
        # t_s_l.append(s)
        # t_a_l.append(a)
        # t_r_l.append(reward(s, a) )
        t_s_l[t, :] = s
        t_a_l[t, :] = a
        t_r_l[t, :] = reward(s, a)
      return t_s_l, t_a_l, t_r_l
    
    value_ester = ValueEster(s_len)
    for i in range(100*40):
      t_s_l, t_a_l, t_r_l = gen_traj()
      scher.train_w_single_traj(t_s_l, t_a_l, t_r_l)
      # value_ester.train_w_single_traj(t_s_l, t_r_l)
      if i % 10 == 0:
        evaluate()
  
  def train_w_mult_trajs():
    N, T = 100, 10
    def gen_N_traj():
      n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((N, T, s_len)), np.zeros((N, T)), np.zeros((N, T))
      for n in range(N):
        for t in range(T):
          s = state()
          a = scher.get_random_action(s)
          n_t_s_l[n, t, :] = s
          n_t_a_l[n, t] = a
          n_t_r_l[n, t] = reward(s, a)
      return n_t_s_l, n_t_a_l, n_t_r_l
      
    for i in range(100*20):
      n_t_s_l, n_t_a_l, n_t_r_l = gen_N_traj()
      scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      if i % 10 == 0:
        evaluate()
  if straj_training:
    train_w_single_traj()
  else:
    train_w_mult_trajs()

def vsimple_regress():
  s_len = 3
  T = 100
  def state():
    s = np.random.randint(10, size=s_len)
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def reward(s):
    return 10*max(s)
  
  def gen_traj():
    t_s_l, t_r_l = np.zeros((T, s_len)), np.zeros((T, 1))
    for t in range(T):
      s = state()
      t_s_l[t, :] = s
      t_r_l[t, :] = reward(s)
    return t_s_l, t_r_l
  
  value_ester = ValueEster(s_len)
  for i in range(100*40):
    t_s_l, t_r_l = gen_traj()
    value_ester.train_w_single_traj(t_s_l, t_r_l)

if __name__ == "__main__":
  test()
  # vsimple_regress()
  