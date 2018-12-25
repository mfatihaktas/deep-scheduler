from rlearning import *

# ##########################################  Explorer  ########################################## #
class Explorer():
  def __init__(self, a_len):
    self.a_len = a_len
  
  def refine(self):
    pass

class EpsGreedyExplorer(Explorer):
  def __init__(self, a_len):
    super().__init__(a_len)
    
    self.eps = 0.1
  
  def __repr__(self):
    return 'EpsGreedyExplorer'
  
  def get_action(self, s, a_q_l):
    if random.uniform(0, 1) < self.eps:
      return np.random.randint(self.a_len, size=1)[0]
    else:
      return np.argmax(a_q_l)
  
  def refine(self):
    self.eps *= 0.99
    blog(eps=self.eps)

class SoftMaxExplorer(Explorer):
  def __init__(self, a_len):
    super().__init__(a_len)
  
  def __repr__(self):
    return 'SoftMaxExplorer'
  
  def get_action(self, s, a_q_l):
    ## Softmax with temperature parameter equal to 1
    try:
      a_q_l /= sum(a_q_l)
      
      a_l = list(range(self.a_len) )
      s = sum([math.exp(a_q_l[a] ) for a in a_l] )
      p_l = [math.exp(a_q_l[a] )/s for a in a_l]
      dist = scipy.stats.rv_discrete(values=(a_l, p_l) )
      return dist.rvs(size=1)[0]
    except:
      return np.argmax(a_q_l)

class UCBExplorer(Explorer):
  def __init__(self, a_len):
    super().__init__(a_len)
    
    self.s_a_nvisit_m = {}
    
    self.jtotaldemand_step = 5
    self.wload_step = 0.1
    
    self.a_for_uncertain_q = Queue(1000)
  
  def __repr__(self):
    return 'UCBExplorer'
  
  def refine(self):
    # blog(s_a_nvisit_m=self.s_a_nvisit_m)
    blog(frac_a_for_uncertain=np.mean(self.a_for_uncertain_q.l) )
  
  def discretize_state(self, s):
    jtotaldemand = int(math.floor(s[0]/self.jtotaldemand_step)*self.jtotaldemand_step)
    cluster_qlen = s[1]
    if STATE_LEN == 3:
      mean_wload = round(math.floor(s[2]/self.wload_step)*self.wload_step, 1)
      return (jtotaldemand, cluster_qlen, mean_wload)
    elif STATE_LEN == 4:
      mean_wload = round(math.floor(s[2]/self.wload_step)*self.wload_step, 1)
      std_wload = round(math.floor(s[3]/self.wload_step)*self.wload_step, 1)
      return (jtotaldemand, cluster_qlen, mean_wload, std_wload)
    elif STATE_LEN == 6:
      min_wload = round(math.floor(s[2]/self.wload_step)*self.wload_step, 1)
      max_wload = round(math.floor(s[3]/self.wload_step)*self.wload_step, 1)
      mean_wload = round(math.floor(s[4]/self.wload_step)*self.wload_step, 1)
      std_wload = round(math.floor(s[5]/self.wload_step)*self.wload_step, 1)
      return (jtotaldemand, cluster_qlen, min_wload, max_wload, mean_wload, std_wload)
    else:
      log(ERROR, "unrecognized STATE_LEN= {}".format(STATE_LEN) )
  
  def get_action(self, s, a_q_l):
    disc_s = self.discretize_state(s)
    if disc_s not in self.s_a_nvisit_m:
      self.s_a_nvisit_m[disc_s] = {a: 0 for a in range(self.a_len) }
    a_nvisit_m = self.s_a_nvisit_m[disc_s]
    # log(INFO, "", s_a_nvisit_m=self.s_a_nvisit_m)
    
    total_nvisit = 0
    for a, nvisit in a_nvisit_m.items():
      if nvisit == 0:
        a_nvisit_m[a] += 1
        self.a_for_uncertain_q.put(1)
        return a
      total_nvisit += nvisit
    
    _a = np.argmax(a_q_l)
    for a, nvisit in a_nvisit_m.items():
      a_q_l[a] += 2*math.sqrt(2*math.log(total_nvisit)/nvisit) # 10*
    
    a = np.argmax(a_q_l)
    a_nvisit_m[a] += 1
    if a == _a:
      self.a_for_uncertain_q.put(0)
    else:
      self.a_for_uncertain_q.put(1)
    return a

# ###########################################  Q Learning  ####################################### #
class QLearner(Learner):
  def __init__(self, s_len, a_len, nn_len=10, save_dir='save'):
    super().__init__(s_len, a_len, nn_len, save_dir)
    self.init()
    # self.explorer = EpsGreedyExplorer(a_len)
    self.explorer = UCBExplorer(a_len)
    self.saver = tf.train.Saver(max_to_keep=2)
    
  def __repr__(self):
    return 'QLearner(s_len= {}, a_len= {}, explorer= {})'.format(self.s_len, self.a_len, self.explorer)
  
  def init(self):
    # N x T x s_len
    self.s_ph = tf.placeholder(tf.float32, shape=(None, None, self.s_len) )
    hidden1 = tf.contrib.layers.fully_connected(self.s_ph, self.nn_len, activation_fn=tf.nn.relu)
    hidden2 = tf.contrib.layers.fully_connected(hidden1, self.nn_len, activation_fn=tf.nn.relu)
    self.Qa_ph = tf.contrib.layers.fully_connected(hidden2, self.a_len, activation_fn=None)
    
    self.a_ph = tf.placeholder(tf.int32, shape=(None, None, 1), name='a_ph')
    self.targetq_ph = tf.placeholder(tf.float32, shape=(None, None, 1), name='q_ph')
    
    sh = tf.shape(self.Qa_ph)
    N, T = sh[0], sh[1]
    indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.a_ph, [-1] )
    self.resp_outputs = tf.reshape(tf.gather(tf.reshape(self.Qa_ph, [-1] ), indices), (N, T, 1) )
    self.loss = tf.losses.mean_squared_error(self.resp_outputs, self.targetq_ph)
    
    self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    self.train_op = self.optimizer.minimize(self.loss)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
  
  def end_of_train(self, loss):
    self.explorer.refine()
    self.num_training += 1
    log(INFO, "{}:: loss= {}".format(self.__class__.__name__, loss), num_training=self.num_training)
  
  def train_w_sarsa_l(self, sarsa_l):
    if len(sarsa_l) == 0:
      log(WARNING, "sarsa_l is empty, skipping.")
      return
    t_s_l, t_a_l, t_targetq_l = [], [], []
    for sarsa in sarsa_l:
      s, a, r, snext = sarsa[0], sarsa[1], sarsa[2], sarsa[3]
      t_s_l.append(s)
      t_a_l.append(a)
      
      a_q_l = self.sess.run(self.Qa_ph,
                            feed_dict={self.s_ph: [[snext]] } )[0][0]
      targetq = r + self.gamma*max(a_q_l)
      t_targetq_l.append(targetq)
    # blog(targetq_l=targetq_l)
    
    loss, _ = self.sess.run([self.q_net.loss, self.q_net.train_op],
                            feed_dict={self.q_net.s_ph: [t_s_l],
                                       self.q_net.a_ph: [t_a_l],
                                       self.q_net.targetq_ph: [t_targetq_l] } )
    self.end_of_train(loss)
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    n_t_q_l = self.sess.run(self.Qa_ph,
                            feed_dict={self.s_ph: n_t_s_l} )
    n_t_targetq_l = np.zeros((N, T, 1))
    for n in range(N):
      for t in range(T):
        if t < T-1:
          n_t_targetq_l[n, t, 0] = n_t_r_l[n, t, 0] + self.gamma*max(n_t_q_l[n, t+1, :] )
        else:
          n_t_targetq_l[n, t, 0] = max(n_t_q_l[n, t, :] )
    
    # n_t_targetq_l = np.zeros((N, T, 1))
    # for n in range(N):
    #   n_t_targetq_l[n] = rewards_to_qvals(n_t_r_l[n], self.gamma)
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                       self.a_ph: n_t_a_l,
                                       self.targetq_ph: n_t_targetq_l} )
    self.end_of_train(loss)
  
  def train_w_mult_trajs_(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    n_t_q_l = self.sess.run(self.Qa_ph,
                            feed_dict={self.s_ph: n_t_s_l} )
    def target_q_w_mstep(m):
      n_t_targetq_l = np.zeros((N, T, 1))
      for n in range(N):
        for t in range(T):
          if t < T-1:
            cumr = 0
            tu_ = min(T-2, t+m-1)
            for t_ in range(tu_, t-1, -1):
              cumr = n_t_r_l[n, t_, 0] + self.gamma*cumr
            n_t_targetq_l[n, t, 0] = cumr + self.gamma*max(n_t_q_l[n, tu_+1, :] )
            # n_t_targetq_l[n, t, 0] = cumr + self.gamma*n_t_q_l[n, tu_+1, n_t_a_l[n, tu_+1, 0] ] # SARSA
          else:
            n_t_targetq_l[n, t, 0] = max(n_t_q_l[n, t, :] )
            # n_t_targetq_l[n, t, 0] = n_t_q_l[n, t, n_t_a_l[n, t, 0] ] # SARSA
      return n_t_targetq_l
    
    lambda_, L = 0.5, 10
    n_t_targetq_l = np.zeros((N, T, 1))
    for m in range(1, L):
      n_t_targetq_l += lambda_**(m-1) * target_q_w_mstep(m)
    n_t_targetq_l *= (1 - lambda_)/(1 - lambda_**L)
    
    loss, _ = self.sess.run([self.loss, self.train_op],
                            feed_dict={self.s_ph: n_t_s_l,
                                       self.a_ph: n_t_a_l,
                                       self.targetq_ph: n_t_targetq_l} )
    self.end_of_train(loss)
  
  def get_random_action(self, s):
    a_q_l = self.sess.run(self.q_net.Qa_ph,
                          feed_dict={self.q_net.s_ph: [[s]] } )[0][0]
    return self.explorer.get_action(s, a_q_l)
  
  def get_max_action(self, s):
    a_q_l = self.sess.run(self.Qa_ph,
                         feed_dict={self.s_ph: [[s]] } )[0][0]
    return np.argmax(a_q_l)
  
  def get_a_q_l(self, s):
    a_q_l = self.sess.run(self.Qa_ph,
                         feed_dict={self.s_ph: [[s]] } )[0][0]
    return a_q_l

# ######################################  QLearner_wTargetNet  ################################### #
class DQNNet:
  def __init__(self, name, s_len, a_len, nn_len):
    self.name = name
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    
    with tf.variable_scope(name):
      # N x T x s_len
      self.s_ph = tf.placeholder(tf.float32, shape=(None, None, s_len) )
      # hidden1 = tf.contrib.layers.fully_connected(self.s_ph, nn_len, activation_fn=tf.nn.relu)
      # hidden2 = tf.contrib.layers.fully_connected(hidden1, nn_len, activation_fn=tf.nn.relu)
      # self.Qa_ph = tf.contrib.layers.fully_connected(hidden2, a_len, activation_fn=None)
      hidden1 = tf.contrib.layers.fully_connected(self.s_ph, nn_len, activation_fn=tf.nn.relu)
      hidden2 = tf.contrib.layers.fully_connected(hidden1, nn_len, activation_fn=tf.nn.relu)
      hidden3 = tf.contrib.layers.fully_connected(hidden2, nn_len, activation_fn=tf.nn.relu)
      self.Qa_ph = tf.contrib.layers.fully_connected(hidden3, a_len, activation_fn=None)
      
      self.a_ph = tf.placeholder(tf.int32, shape=(None, None, 1), name="a_ph")
      self.targetq_ph = tf.placeholder(tf.float32, shape=(None, None, 1), name="q_ph")
      
      sh = tf.shape(self.Qa_ph)
      N, T = sh[0], sh[1]
      indices = tf.range(0, N*T)*sh[2] + tf.reshape(self.a_ph, [-1] )
      self.resp_outputs = tf.reshape(tf.gather(tf.reshape(self.Qa_ph, [-1] ), indices), (N, T, 1) )
      # self.loss = tf.losses.mean_squared_error(self.resp_outputs, self.targetq_ph)
      self.loss = tf.losses.huber_loss(self.resp_outputs, self.targetq_ph)
      
      self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
      self.train_op = self.optimizer.minimize(self.loss)
      
  def __repr__(self):
    return 'DQNNet(name= {}, s_len= {}, a_len= {})'.format(self.name, self.s_len, self.a_len)

NUM_TRAINING_BEFORE_QNET_TO_TARGET = 5
class QLearner_wTargetNet(Learner):
  def __init__(self, s_len, a_len, nn_len=10, save_dir='save'):
    super().__init__(s_len, a_len, nn_len, save_dir)
    self.q_net = DQNNet('QNet', s_len, a_len, nn_len)
    self.target_net = DQNNet('TargetNet', s_len, a_len, nn_len)
    self.num_training = 0
    # self.explorer = EpsGreedyExplorer(a_len)
    self.explorer = UCBExplorer(a_len)
    
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer() )
    self.saver = tf.train.Saver(max_to_keep=2)
  
  def __repr__(self):
    return 'QLearner_wTargetNet(s_len= {}, a_len= {}, explorer= {})'.format(self.s_len, self.a_len, self.explorer)
  
  def update_target_net(self):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'QNet')
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'TargetNet')
  
    op_holder = []
    # Update TargetNet parameters with QNet parameters
    for from_var, to_var in zip(from_vars, to_vars):
      op_holder.append(to_var.assign(from_var) )
    self.sess.run(op_holder)
    log(INFO, "<< updated TargetNet with QNet >>\n")
  
  def end_of_train(self, loss):
    self.explorer.refine()
    self.num_training += 1
    log(INFO, "{}:: loss= {}".format(self.__class__.__name__, loss), num_training=self.num_training)
    if self.num_training % NUM_TRAINING_BEFORE_QNET_TO_TARGET == 0:
      self.update_target_net()
  
  def train_w_sarsa_l(self, sarsa_l):
    T = len(sarsa_l)
    if len(sarsa_l) == 0:
      log(WARNING, "sarsa_l is empty, skipping.")
      return
    t_s_l, t_a_l, t_targetq_l = np.zeros((T, self.s_len)), np.zeros((T, 1)), np.zeros((T, 1))
    for t, sarsa in enumerate(sarsa_l):
      s, a, r, snext = sarsa[0], sarsa[1], sarsa[2], sarsa[3]
      t_s_l[t, :] = s
      t_a_l[t, :] = a
      
      a_q_l = self.sess.run(self.target_net.Qa_ph,
                            feed_dict={self.target_net.s_ph: [[snext]] } )[0][0]
      t_targetq_l[t, :] = r + self.gamma*max(a_q_l) # Q-learning
      # t_targetq_l[t, :] = r + self.gamma*a_q_l[sarsa[4] ] # SARSA
    
    loss, _ = self.sess.run([self.q_net.loss, self.q_net.train_op],
                            feed_dict={self.q_net.s_ph: [t_s_l],
                                       self.q_net.a_ph: [t_a_l],
                                       self.q_net.targetq_ph: [t_targetq_l] } )
    self.end_of_train(loss)
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    n_t_q_l = self.sess.run(self.target_net.Qa_ph,
                            feed_dict={self.target_net.s_ph: n_t_s_l} )
    n_t_targetq_l = np.zeros((N, T, 1))
    for n in range(N):
      for t in range(T):
        if t < T-1:
          n_t_targetq_l[n, t, 0] = n_t_r_l[n, t, 0] + self.gamma*max(n_t_q_l[n, t+1, :] )
        else:
          n_t_targetq_l[n, t, 0] = max(n_t_q_l[n, t, :] )
    
    loss, _ = self.sess.run([self.q_net.loss, self.q_net.train_op],
                            feed_dict={self.q_net.s_ph: n_t_s_l,
                                       self.q_net.a_ph: n_t_a_l,
                                       self.q_net.targetq_ph: n_t_targetq_l} )
    self.end_of_train(loss)
  
  def train_w_mult_trajs_(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    n_t_q_l = self.sess.run(self.target_net.Qa_ph,
                            feed_dict={self.target_net.s_ph: n_t_s_l} )
    def target_q_w_mstep(m):
      n_t_targetq_l = np.zeros((N, T, 1))
      for n in range(N):
        for t in range(T):
          if t < T-1:
            cumr = 0
            tu_ = min(T-2, t+m-1)
            for t_ in range(tu_, t-1, -1):
              cumr = n_t_r_l[n, t_, 0] + self.gamma*cumr
            n_t_targetq_l[n, t, 0] = cumr + self.gamma*max(n_t_q_l[n, tu_+1, :] )
            # n_t_targetq_l[n, t, 0] = cumr + self.gamma*n_t_q_l[n, tu_+1, n_t_a_l[n, tu_+1, 0] ] # SARSA
          else:
            n_t_targetq_l[n, t, 0] = max(n_t_q_l[n, t, :] )
            # n_t_targetq_l[n, t, 0] = n_t_q_l[n, t, n_t_a_l[n, t, 0] ] # SARSA
      return n_t_targetq_l
    
    lambda_, L = 0.5, 10
    n_t_targetq_l = np.zeros((N, T, 1))
    for m in range(1, L):
      n_t_targetq_l += lambda_**(m-1) * target_q_w_mstep(m)
    n_t_targetq_l *= (1 - lambda_)/(1 - lambda_**L)
    
    loss, _ = self.sess.run([self.q_net.loss, self.q_net.train_op],
                            feed_dict={self.q_net.s_ph: n_t_s_l,
                                       self.q_net.a_ph: n_t_a_l,
                                       self.q_net.targetq_ph: n_t_targetq_l} )
    self.end_of_train(loss)
  
  def get_random_action(self, s):
    a_q_l = self.sess.run(self.q_net.Qa_ph,
                          feed_dict={self.q_net.s_ph: [[s]] } )[0][0]
    return self.explorer.get_action(s, a_q_l)
  
  def get_max_action(self, s):
    a_q_l = self.sess.run(self.q_net.Qa_ph,
                          feed_dict={self.q_net.s_ph: [[s]] } )[0][0]
    return np.argmax(a_q_l)
  
  def get_a_q_l(self, s):
    a_q_l = self.sess.run(self.q_net.Qa_ph,
                         feed_dict={self.q_net.s_ph: [[s]] } )[0][0]
    return a_q_l

# ################################  QLearner_wTargetNet_wExpReplay  ############################## #
class Queue(object):
  def __init__(self, size):
    self.size = size
    
    self.l = []
  
  def put(self, e):
    if len(self.l) == self.size:
      self.l.pop(0)
    self.l.append(e)
  
  def put_l(self, e_l):
    if len(self.l) + len(e_l) >= self.size:
        self.l[0:(len(e_l) + len(self.l)) - self.size] = []
    self.l.extend(e_l)
  
class ExpQueue(Queue):
  def __init__(self, buffer_size, batch_size):
    super().__init__(buffer_size)
    self.buffer_size = buffer_size
    self.batch_size = batch_size
  
  def __repr__(self):
    return 'ExpQueue(buffer_size= {}, batch_size= {})'.format(self.buffer_size, self.batch_size)
  
  def sample_batch(self):
    try:
      # return random.sample(list(range(len(self.l) ) ), self.batch_size)
      return random.sample(self.l, self.batch_size)
    except ValueError:
      return []

class QLearner_wTargetNet_wExpReplay(QLearner_wTargetNet):
  def __init__(self, s_len, a_len, exp_buffer_size, exp_batch_size, nn_len=10, save_dir='save'):
    super().__init__(s_len, a_len, nn_len, save_dir)
    self.exp_buffer_size = exp_buffer_size
    self.exp_batch_size = exp_batch_size
    
    self.exp_q = ExpQueue(exp_buffer_size, exp_batch_size)
    self.learning_count = 0
  
  def __repr__(self):
    return 'QLearner_wTargetNet_wExpReplay(s_len= {}, a_len= {}, exp_q= {}, explorer= {})'.format(self.s_len, self.a_len, self.exp_q, self.explorer)
  
  def train_w_mult_trajs(self, n_t_s_l, n_t_a_l, n_t_r_l):
    N = len(n_t_s_l)
    T = len(n_t_s_l[0] )
    
    sarsa_l = []
    for n in range(N):
      for t in range(T-1):
        s, a, r = n_t_s_l[n, t], n_t_a_l[n, t], n_t_r_l[n, t]
        snext, anext = n_t_s_l[n, t+1], n_t_a_l[n, t+1]
        sarsa_l.append((s, a, r, snext, anext) )
    # blog(sarsa_l=sarsa_l)
    self.exp_q.put_l(sarsa_l)
    
    sarsa_l = []
    for _ in range(10):
      sarsa_l.extend(self.exp_q.sample_batch() )
    self.train_w_sarsa_l(sarsa_l)
  
