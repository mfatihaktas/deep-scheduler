import math, time, random
import numpy as np
import tensorflow as tf

class DeepScher(object):
  def __init__(self, state_len, action_len, nn_len=10):
    self.state_len = state_len
    self.action_len = action_len
    self.nn_len = nn_len
    
    self.gamma = 0.75
    self.init()
  
  def __repr__(self):
    return "DeepScher[state_len= {}, action_len= {}]".format(self.state_len, self.action_len)
  
  def init(self):
    # self.state_ph = tf.placeholder(tf.float32, shape=(None, self.state_len) )
    # N x T x state_len
    self.state_ph = tf.placeholder(tf.float32, shape=(None, None, self.state_len) )
    with tf.name_scope('hidden1'):
      w = tf.Variable(
            tf.truncated_normal([1, self.state_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.state_len) ) ),
            name='weights')
      b = tf.Variable(tf.zeros([1, self.nn_len] ), name='biases')
      hidden1 = tf.nn.relu(tf.matmul(self.state_ph, w) + b)
    with tf.name_scope('hidden2'):
      w = tf.Variable(
            tf.truncated_normal([1, self.nn_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
            name='weights')
      b = tf.Variable(tf.zeros([1, self.nn_len] ), name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
    with tf.name_scope('action_probs'):
      w = tf.Variable(
          tf.truncated_normal([1, self.nn_len, self.action_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
          name='weights')
      b = tf.Variable(tf.zeros([1, self.action_len] ), name='biases')
      self.action_probs = tf.nn.softmax(tf.matmul(hidden2, w) + b)
    hidden1 = tf.contrib.slim.fully_connected(self.state_ph, self.nn_len, activation_fn=tf.nn.relu)
    hidden2 = tf.contrib.slim.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
    self.action_probs = tf.contrib.slim.fully_connected(hidden2, self.action_len, activation_fn=tf.nn.softmax)
    
    # For training with single trajectory
    # self.action_ph = tf.placeholder(shape=[None], dtype=tf.int32)
    # self.reward_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    # self.baseline_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    # sh = tf.shape(self.action_probs)
    # indexes = tf.range(0, sh[0] )*sh[1] + self.action_ph
    # self.responsible_outputs = tf.gather(tf.reshape(self.action_probs, [-1]), indexes)
    # self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*(self.reward_ph - self.baseline_ph) )
    
    # For training with multiple trajectories
    self.action_ph = tf.placeholder(shape=[None, None], dtype=tf.int32)
    self.reward_ph = tf.placeholder(shape=[None, None], dtype=tf.float32)
    self.baseline_ph = tf.placeholder(shape=[None, None], dtype=tf.float32)
    
    sh = tf.shape(self.action_probs)
    N, T = sh[0], sh[1]
    indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.action_ph, [-1] )
    self.responsible_outputs = tf.reshape(tf.gather(tf.reshape(self.action_probs, [-1] ), indices), (sh[0], sh[1] ) )
    # self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*(self.reward_ph - self.baseline_ph) )
    self.loss = -tf.reduce_sum(tf.reduce_mean(tf.log(self.responsible_outputs)*(self.reward_ph - self.baseline_ph), axis=0) )
    
    # self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    self.optimizer = tf.train.AdamOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def get_random_action(self, s):
    # print("s= {}".format(s) )
    action_probs = self.sess.run(self.action_probs, feed_dict={self.state_ph: [[s]] } )
    a_dist = np.array(action_probs[0][0] )
    # print("a_dist= {}".format(a_dist) )
    a = np.random.choice(a_dist, 1, p=a_dist)
    a = np.argmax(a_dist == a)
    # print("a= {}".format(a) )
    return a
  
  def get_max_action(self, s):
    action_probs = self.sess.run(self.action_probs, feed_dict={self.state_ph: [[s]] } )
    return np.argmax(action_probs[0][0] )
  
  def discount_rewards(self, r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r) ) ):
      running_add = r[t] + running_add * self.gamma
      discounted_r[t] = running_add
    return discounted_r
  
  def train_by_one_sample_at_atime(self, s_l, a_l, r_l):
    r_l = self.discount_rewards(r_l)
    base = sum(r_l)/len(r_l)
    for i, s in enumerate(s_l):
      loss, _ = self.sess.run([self.loss, self.train_op],
                              feed_dict={self.state_ph: [s],
                                         self.action_ph: [a_l[i]],
                                         self.reward_ph: [r_l[i]],
                                         self.baseline_ph: [base] } )
  
  def train_w_single_traj(self, s_l, a_l, r_l):
    # print("r_l= {}".format(r_l) )
    r_l = self.discount_rewards(r_l)
    base = sum(r_l)/len(r_l)
    # print("discounted; r_l= {}".format(r_l) )
    # s = np.reshape(s_l, (len(s_l), self.state_len) )
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.state_ph: [s_l],
                                       self.action_ph: [a_l],
                                       self.reward_ph: [r_l],
                                       self.baseline_ph: [[base]*len(r_l)] } )
    # print("loss= {}".format(loss) )
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    # All trajectories use the same policy
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    for n in range(N):
      n_t_r_l[n] = self.discount_rewards(n_t_r_l[n] )
    t_avgr_l = [np.mean(n_t_r_l[:, t] ) for t in range(T) ]
    n_t_b_l = np.array([t_avgr_l for n in range(N) ] )
    # n_t_b_l = np.array([np.mean(n_t_r_l)] * N*T).reshape((N, T) )
    
    # print("n_t_a_l= {}".format(n_t_a_l) )
    # print("n_t_r_l= {}".format(n_t_r_l) )
    # print("n_t_b_l= {}".format(n_t_b_l) )
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.state_ph: n_t_s_l,
                                      self.action_ph: n_t_a_l,
                                      self.reward_ph: n_t_r_l,
                                      self.baseline_ph: n_t_b_l} )
    # print("sh= {}".format(sh) )
    # print(">> loss= {}".format(loss) )
  
def test():
  s_len, a_len, nn_len = 3, 3, 10
  scher = DeepScher(s_len, a_len, nn_len)
  
  def state():
    s = np.random.randint(10, size=s_len)
    sum_s = sum(np.random.randint(10, size=s_len) )
    return s/sum_s if sum_s != 0 else s
  
  def reward(s, a):
    # s_min = min(s)
    # r = 10 if s[a] == s_min else 0
    # return min(100, 1/(0.001 + s[a] - min(s) ) )
    # return 100*math.exp(-(s[a] - min(s) ) )
    return 1/(0.001 + s[a] - min(s) )
    
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
      s_l, a_l, r_l = [], [], []
      for t in range(T):
        s = state()
        a = scher.get_random_action(s)
        s_l.append(s)
        a_l.append(a)
        r_l.append(reward(s, a) )
      return s_l, a_l, r_l
    
    for i in range(100*20):
      s_l, a_l, r_l = gen_traj()
      scher.train_w_single_traj(s_l, a_l, r_l)
      # scher.train_by_one_sample_at_atime(s_l, action_l, reward_l)
      if i % 100 == 0:
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
  # train_w_single_traj()
  train_w_mult_trajs()

if __name__ == "__main__":
  test()
  