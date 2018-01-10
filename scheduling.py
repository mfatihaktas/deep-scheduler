import math, time, random
import numpy as np
import tensorflow as tf

class DeepScheduler(object):
  def __init__(self, state_len, action_len, nn_len):
    self.state_len = state_len
    self.action_len = action_len
    self.nn_len = nn_len
    
    self.gamma = 0.1
    self.init()
  
  def init(self):
    self.state_input_ph = tf.placeholder(tf.float32, shape=(None, self.state_len) )
    with tf.name_scope('hidden1'):
      w = tf.Variable(
            tf.truncated_normal([self.state_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.state_len) ) ),
            name='weights')
      tf.summary.histogram('histogram', w)
      b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      hidden1 = tf.nn.relu(tf.matmul(self.state_input_ph, w) + b)
    with tf.name_scope('hidden2'):
      w = tf.Variable(
            tf.truncated_normal([self.nn_len, self.nn_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
            name='weights')
      tf.summary.histogram('histogram', w)
      b = tf.Variable(tf.zeros([self.nn_len] ), name='biases')
      hidden2 = tf.nn.relu(tf.matmul(hidden1, w) + b)
    with tf.name_scope('action_probs'):
      w = tf.Variable(
          tf.truncated_normal([self.nn_len, self.action_len], stddev=1.0 / math.sqrt(float(self.nn_len) ) ),
          name='weights')
      tf.summary.histogram('histogram', w)
      b = tf.Variable(tf.zeros([self.action_len] ), name='biases')
      self.action_probs = tf.nn.softmax(tf.matmul(hidden2, w) + b)
      # self.chosen_action = tf.argmax(self.action_probs, 1)
    
    self.action_ph = tf.placeholder(shape=[None], dtype=tf.int32)
    self.reward_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    self.reward_base_ph = tf.placeholder(shape=[None], dtype=tf.float32)
    
    self.indexes = tf.range(0, tf.shape(self.action_probs)[0]) * tf.shape(self.action_probs)[1] + self.action_ph
    self.responsible_outputs = tf.gather(tf.reshape(self.action_probs, [-1]), self.indexes)
    self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*(self.reward_ph - self.reward_base_ph) )
    
    self.optimizer = tf.train.GradientDescentOptimizer(0.01)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def discount_rewards(self, r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r) ) ):
      running_add = running_add * self.gamma + r[t]
      discounted_r[t] = running_add
    return discounted_r
  
  # Train with a single trajectory
  def train(self, state_l, action_l, reward_l):
    # print("reward_l= {}".format(reward_l) )
    reward_l = self.discount_rewards(reward_l)
    base = sum(reward_l)/len(reward_l)
    # print("discounted; reward_l= {}".format(reward_l) )
    state_input = np.reshape(state_l, (len(state_l), self.state_len) )
    # print("state_input= {}".format(state_input) )
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.state_input_ph: state_input,
                                       self.action_ph: action_l,
                                       self.reward_ph: reward_l,
                                       self.reward_base_ph: [base]*len(reward_l) } )
    # print("loss= {}".format(loss) )
  
  def get_random_action(self, state):
    # print("state= {}".format(state) )
    state_input = np.reshape(state, (1, self.state_len) )
    action_probs = self.sess.run(self.action_probs, feed_dict={self.state_input_ph: state_input} )
    a_dist = np.array(action_probs[0] )
    # print("a_dist= {}".format(a_dist) )
    a = np.random.choice(a_dist, 1, p=a_dist)
    a = np.argmax(a_dist == a)
    # 
    # print("a= {}".format(a) )
    return a
  
  def get_max_action(self, state):
    state_input = np.reshape(state, (1, self.state_len) )
    action_probs = self.sess.run(self.action_probs, feed_dict={self.state_input_ph: state_input} )
    return np.argmax(action_probs[0] )

def test_scheduler():
  nn_len = 10
  state_len, action_len = 3, 3
  scher = DeepScheduler(state_len, action_len, nn_len)
  
  def state():
    s = np.random.randint(10, size=state_len)
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def reward(s, a):
    # s_min = min(s)
    # r = 10 if s[a] == s_min else 0
    return min(100, 1/(0.001 + s[a] - min(s) ) )
    # return math.exp(-(s[a] - min(s) ) )
  
  T = 100
  for i in range(100*20):
    state_l, action_l, reward_l = [], [], []
    for t in range(T):
      s = state()
      state_l.append(s)
      
      a = scher.get_random_action(s)
      action_l.append(a)
      
      reward_l.append(reward(s, a) )
    # print("i= {}, reward_l= {}".format(i, reward_l) )
    scher.train(state_l, action_l, reward_l)
    
    if i % 100 == 0:
      num_shortest_found = 0
      for e in range(100):
        s = state()
        a = scher.get_max_action(s)
        
        if s[a] - min(s) < 0.01:
          num_shortest_found += 1
      print("freq shortest found= {}".format(num_shortest_found/100) )

if __name__ == "__main__":
  test_scheduler()
  