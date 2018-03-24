import math, time, random
import numpy as np
import tensorflow as tf

def rewards_to_qvals(t_r_l, gamma):
  T = t_r_l.shape[0]
  # reward = average of all following rewards
  # for t in range(T):
  #   t_r_l[t, 0] = np.mean(t_r_l[t:, 0])
  for t in range(T):
    w_l, r_l = [], []
    for i, r in enumerate(t_r_l[t:, 0] ):
      w_l.append(gamma**i)
      r_l.append(gamma**i * r)
    t_r_l[t, 0] = sum(r_l)/sum(w_l)
  return t_r_l
  
  # reward = discounted sum of all following rewards
  # t_dr_l = np.zeros((T, 1))
  # radd = 0
  # for t in range(T-1, -1, -1):
  #   radd = t_r_l[t, 0] + radd * gamma
  #   t_dr_l[t, 0] = radd
  # return t_dr_l

class ValueEster(object):
  def __init__(self, s_len, nn_len, straj_training):
    self.s_len = s_len
    self.nn_len = nn_len
    self.straj_training = straj_training
    
    self.gamma = 0.99
    self.init()
  
  def __repr__(self):
    return "ValueEster[s_len= {}]".format(self.s_len)
  
  def init(self):
    if self.straj_training:
      # T x s_len
      self.s_ph = tf.placeholder(shape=(None, self.s_len), dtype=tf.float32)
      # with tf.name_scope('hidden1'):
      #   w = tf.Variable(
      #         tf.truncated_normal([self.s_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.s_len) ) ),
      #         name='weights')
      #   b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      #   hidden1 = tf.nn.relu(tf.matmul(self.s_ph, w) + b)
      # with tf.name_scope('hidden2'):
      #   w = tf.Variable(
      #         tf.truncated_normal([self.nn_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
      #         name='weights')
      #   b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      #   hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
      # with tf.name_scope('v_layer'):
      #   w = tf.Variable(
      #       tf.truncated_normal([self.nn_len, 1], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
      #       name='weights')
      #   b = tf.Variable(tf.zeros([1] ), name='biases')
      #   self.v = tf.matmul(hidden2, w) + b
      self.hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      self.hidden2 = tf.contrib.layers.fully_connected(self.hidden1, self.nn_len, activation_fn=tf.nn.relu)
      self.v = tf.contrib.layers.fully_connected(self.hidden2, 1, activation_fn=None)
      
      self.sampled_v = tf.placeholder(shape=(None, 1), dtype=tf.float32)
      # self.loss = tf.reduce_sum(tf.squared_difference(self.v, self.sampled_v) )
      self.loss = tf.losses.mean_squared_error(self.v, self.sampled_v)
    else:
      # N x T x s_len
      self.s_ph = tf.placeholder(shape=(None, None, self.s_len), dtype=tf.float32)
      self.hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      self.hidden2 = tf.contrib.layers.fully_connected(self.hidden1, self.nn_len, activation_fn=tf.nn.relu)
      self.v = tf.contrib.layers.fully_connected(self.hidden2, 1, activation_fn=None)
      
      self.sampled_v = tf.placeholder(shape=(None, None, 1), dtype=tf.float32)
      # self.loss = tf.reduce_sum(tf.squared_difference(self.v, self.sampled_v) )
      self.loss = tf.losses.mean_squared_error(self.v, self.sampled_v)
    # self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    self.optimizer = tf.train.AdamOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def train_w_single_traj(self, t_s_l, t_r_l):
    # r_l keeps one step reward r's not v's
    # print("t_s_l.shape= {}".format(t_s_l.shape) )
    # print("t_r_l.shape= {}".format(t_r_l.shape) )
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
    # print("ValueEster:: loss= {}".format(loss) )
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_v_l):
    _, loss = self.sess.run([self.train_op, self.loss],
                            feed_dict={self.s_ph: n_t_s_l,
                                       self.sampled_v: n_t_v_l} )
    print("ValueEster:: loss= {}".format(loss) )
  
  def get_v(self, n_t_s_l):
    if self.straj_training:
      return self.sess.run(self.v,
                           feed_dict={self.s_ph: n_t_s_l} )
    else:
      return self.sess.run(self.v,
                           feed_dict={self.s_ph: n_t_s_l} )

class PolicyGradScher(object):
  def __init__(self, s_len, a_len, nn_len=10, straj_training=False, save_name='log/policy_grad'):
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    self.straj_training = straj_training
    self.save_name = save_name
    
    # self.v_ester = ValueEster(s_len, nn_len, straj_training)
    self.gamma = 0.9 # 0.99
    self.init()
    
    self.saver = tf.train.Saver(max_to_keep=1)
  
  def __repr__(self):
    return "PolicyGradScher[s_len= {}, a_len= {}]".format(self.s_len, self.a_len)
  
  def save(self, step):
    save_path = self.saver.save(self.sess, self.save_name, global_step=step)
    # print("saved scher to save_path= {}".format(save_path) )
  
  def restore(self, step):
    self.saver.restore(self.sess, '{}-{}'.format(self.save_name, step) )
    # print("restored scher to step= {}".format(step) )
  
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
      # self.s_ph = tf.placeholder(tf.float32, shape=(None, None, self.s_len) )
      # hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      # hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
      # self.a_probs = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=tf.nn.softmax)
      
      # self.a_ph = tf.placeholder(tf.int32, shape=(None, None), name="a_ph")
      # self.q_ph = tf.placeholder(tf.float32, shape=(None, None), name="q_ph")
      # self.v_ph = tf.placeholder(tf.float32, shape=(None, None), name="v_ph")
      
      self.s_ph = tf.placeholder(tf.float32, shape=(None, None, self.s_len) )
      hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
      hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
      self.a_probs = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=tf.nn.softmax)
      
      self.a_ph = tf.placeholder(tf.int32, shape=(None, None, 1), name="a_ph")
      self.q_ph = tf.placeholder(tf.float32, shape=(None, None, 1), name="q_ph")
      self.v_ph = tf.placeholder(tf.float32, shape=(None, None, 1), name="v_ph")
      
      sh = tf.shape(self.a_probs)
      N, T = sh[0], sh[1]
      indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.a_ph, [-1] )
      self.resp_outputs = tf.reshape(tf.gather(tf.reshape(self.a_probs, [-1] ), indices), (sh[0], sh[1], 1) )
      self.loss = -tf.reduce_mean(tf.reduce_sum(tf.log(self.resp_outputs)*(self.q_ph - self.v_ph), axis=1), axis=0)
    
    self.optimizer = tf.train.AdamOptimizer(0.01) # tf.train.GradientDescentOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def train_w_single_traj(self, t_s_l, t_a_l, t_r_l):
    # t_s_l: T x s_len, t_a_l, t_r_l: T x 1
    self.v_ester.train_w_single_traj(t_s_l, t_r_l)
    t_v_l = self.v_ester.get_v(t_s_l)
    t_vp1_l = t_v_l[1:]
    t_q_l = np.add(t_r_l[:-1], np.multiply(self.gamma, t_vp1_l) )
    # print("t_vp1_l= {}".format(t_vp1_l) )
    # print("t_v_l= {}".format(t_v_l) )
    loss, _, sh = self.sess.run([self.loss, self.train_op, tf.shape(self.a_probs) ],
                            feed_dict={self.s_ph: t_s_l[:-1],
                                       self.a_ph: t_a_l[:-1],
                                       self.q_ph: t_q_l,
                                       self.v_ph: t_v_l[:-1] } )
    # print("sh= {}".format(sh) )
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    # All trajectories use the same policy
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    # print("n_t_s_l.shape= {}".format(n_t_s_l.shape) )
    # print("n_t_a_l.shape= {}".format(n_t_a_l.shape) )
    # '''
    # Policy gradient
    n_t_q_l = np.zeros((N, T, 1))
    for n in range(N):
      n_t_q_l[n] = rewards_to_qvals(n_t_r_l[n], self.gamma)
    # print("n_t_q_l= {}".format(n_t_q_l) )
    # print("n_t_q_l.shape= {}".format(n_t_q_l.shape) )
    print("avg q= {}".format(np.mean(n_t_q_l) ) )
    
    t_avgr_l = np.array([np.mean(n_t_q_l[:, t, 0] ) for t in range(T) ] ).reshape((T, 1))
    n_t_v_l = np.zeros((N, T, 1))
    for n in range(N):
      n_t_v_l[n] = t_avgr_l
    # print("n_t_v_l= {}".format(n_t_v_l) )
    # print("n_t_v_l.shape= {}".format(n_t_v_l.shape) )
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                      self.a_ph: n_t_a_l,
                                      self.q_ph: n_t_q_l,
                                      self.v_ph: n_t_v_l} )
    print("PolicyGradScher:: loss= {}".format(loss) )
    # '''
    '''
    # Policy gradient by getting baseline values from actor-critic
    n_t_q_l = np.zeros((N, T, 1))
    for n in range(N):
      n_t_q_l[n] = rewards_to_qvals(n_t_r_l[n], self.gamma)
    
    # for n in range(N):
    #   self.v_ester.train_w_single_traj(n_t_s_l[n], n_t_r_l[n] )
    # t_avgr_l = np.array([np.mean(n_t_q_l[:, t, 0] ) for t in range(T) ] ).reshape((T, 1))
    # n_t_v_l = np.zeros((N, T, 1))
    # for n in range(N):
    #   n_t_v_l[n] = t_avgr_l
    self.v_ester.train_w_mult_trajs(n_t_s_l, n_t_q_l)
    n_t_v_l = self.v_ester.get_v(n_t_s_l)
    
    # print("n_t_q_l= {}".format(n_t_q_l) )
    # print("n_t_q_l.shape= {}".format(n_t_q_l.shape) )
    # print("n_t_v_l= {}".format(n_t_v_l) )
    # print("n_t_v_l.shape= {}".format(n_t_v_l.shape) )
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                       self.a_ph: n_t_a_l,
                                       self.q_ph: n_t_q_l,
                                       self.v_ph: n_t_v_l} )
    print("PolicyGradScher:: loss= {}".format(loss) )
    '''
  
  def get_action_dist(self, s):
    a_probs = self.sess.run(self.a_probs,
                            feed_dict={self.s_ph: [[s]] } )
    return np.array(a_probs[0][0] )
  
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

class QLearningScher(object):
  def __init__(self, s_len, a_len, nn_len=10):
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    
    self.gamma = 0.99
    self.eps = 0.1
    self.init()
    
  def __repr__(self):
    return "QLearningScher[s_len= {}, a_len= {}]".format(self.s_len, self.a_len)
  
  def init(self):
    # N x T x s_len
    self.s_ph = tf.placeholder(tf.float32, shape=(None, None, self.s_len) )
    hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
    self.Qa_ph = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=None)
    
    self.a_ph = tf.placeholder(tf.int32, shape=(None, None, 1), name="a_ph")
    self.targetq_ph = tf.placeholder(tf.float32, shape=(None, None, 1), name="q_ph")
    
    sh = tf.shape(self.Qa_ph)
    N, T = sh[0], sh[1]
    indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.a_ph, [-1] )
    self.resp_outputs = tf.reshape(tf.gather(tf.reshape(self.Qa_ph, [-1] ), indices), (sh[0], sh[1], 1) )
    self.loss = tf.losses.mean_squared_error(self.resp_outputs, self.targetq_ph)
    
    self.optimizer = tf.train.AdamOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    # n_t_q_l = self.sess.run(self.Qa_ph,
    #                         feed_dict={self.s_ph: n_t_s_l} )
    # n_t_targetq_l = np.zeros((N, T, 1))
    # for n in range(N):
    #   for t in range(T):
    #     if t < T-1:
    #       n_t_targetq_l[n, t, 0] = n_t_r_l[n, t, 0] + self.gamma*max(n_t_q_l[n, t+1, :] )
    #     else:
    #       n_t_targetq_l[n, t, 0] = n_t_r_l[n, t, 0]
    
    n_t_targetq_l = np.zeros((N, T, 1))
    for n in range(N):
      n_t_targetq_l[n] = rewards_to_qvals(n_t_r_l[n], self.gamma)
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                       self.a_ph: n_t_a_l,
                                       self.targetq_ph: n_t_targetq_l} )
    print("QLearningScher:: loss= {}".format(loss) )
    # self.eps *= 0.95
  
  def get_random_action(self, s):
    if random.uniform(0, 1) < self.eps:
      return np.random.randint(self.a_len, size=1)[0]
    else:
      qa_l = self.sess.run(self.Qa_ph,
                           feed_dict={self.s_ph: [[s]] } )
      return np.argmax(qa_l)
  
  def get_max_action(self, s):
    qa_l = self.sess.run(self.Qa_ph,
                         feed_dict={self.s_ph: [[s]] } )
    return np.argmax(qa_l)
  
def test():
  s_len, a_len, nn_len = 3, 3, 10
  straj_training = False
  scher = PolicyGradScher(s_len, a_len, nn_len, straj_training)
  # scher = QLearningScher(s_len, a_len, nn_len)
  
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
    T = 10
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
    
    value_ester = ValueEster(s_len, nn_len=10, straj_training=False)
    for i in range(100*40):
      t_s_l, t_a_l, t_r_l = gen_traj()
      scher.train_w_single_traj(t_s_l, t_a_l, t_r_l)
      # value_ester.train_w_single_traj(t_s_l, t_r_l)
      if i % 10 == 0:
        evaluate()
  
  def train_w_mult_trajs():
    N, T = 10, 100
    def gen_N_traj():
      n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        for t in range(T):
          s = state()
          a = scher.get_random_action(s)
          n_t_s_l[n, t, :] = s
          n_t_a_l[n, t, :] = a
          n_t_r_l[n, t, :] = reward(s, a)
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
  
  value_ester = ValueEster(s_len, nn_len=10, straj_training=False)
  for i in range(100*40):
    t_s_l, t_r_l = gen_traj()
    value_ester.train_w_single_traj(t_s_l, t_r_l)

if __name__ == "__main__":
  test()
  # vsimple_regress()
  