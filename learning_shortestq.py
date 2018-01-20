import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scheduling import *

# *************************  Learning shortest-q scheduling on the fly  ************************** #
# Learning from single trajectory at a time ended up having too much variance
class LearningShortestQ_wSingleTraj(object):
  def __init__(self, env, n):
    self.env = env
    self.n = n
    
    slowdown_dist = Dolly() # DUniform(1, 1)
    self.q_l = [FCFS(i, env, slowdown_dist, out=self) for i in range(self.n) ]
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    # 
    self.scher = DeepScher(n, n)
    
    self.training_len = 1000
    self.training_on = False
    self.jid_to_wait_l = []
    
    self.action_correctness_l = []
  
  def reset_training(self, jid):
    log(WARNING, "started with jid= {}".format(jid) )
    self.training_on = True
    self.jid_head_of_training = jid
    self.jid_tail_of_training = jid + self.training_len-1
    self.jid_to_wait_l = list(range(self.jid_head_of_training, self.jid_tail_of_training+1) )
    
    self.action_correctness_l.clear()
  
  def __repr__(self):
    return "LearningShortestQ_wSingleTraj[n= {}]".format(self.n)
  
  def state(self):
    return np.mean(np.array([q.length() for q in self.q_l] ) )
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      # print("s= {}".format(s) )
      a = self.scher.get_random_action(s)
      
      a_ = self.scher.get_max_action(s)
      c = 1 if (s[a_] - min(s) ) < 0.1 else 0
      self.action_correctness_l.append(c)
      
      # print("qid= {}".format(a) )
      if not self.training_on:
        self.reset_training(j._id)
      
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.tsize, 's': s, 'a': a}    
      self.q_l[a].put(Task(j._id, j.k, j.tsize) )
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    if jid not in self.jid_info_m: # from jobs that were not included in training set
      return
    
    jinfo = self.jid_info_m[jid]
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    
    try:
      self.jid_to_wait_l.remove(jid)
      # log(WARNING, "removed jid= {}, len(jid_to_wait_l)= {}".format(jid, len(self.jid_to_wait_l) ) )
    except:
      pass
      # log(WARNING, "could not remove from jid_to_wait_l; jid= {}".format(jid) )
    if len(self.jid_to_wait_l) == 0: # Ready for training
      print("Training:: with jobs from {} to {}".format(self.jid_head_of_training, self.jid_tail_of_training) )
      self.training_on = False
      s_l, a_l, r_l = [], [], []
      for jid in range(self.jid_head_of_training, self.jid_tail_of_training+1):
        jinfo_m = self.jid_info_m[jid]
        s_l.append(jinfo_m['s'] )
        a_l.append(jinfo_m['a'] )
        r_l.append(1/jinfo_m['sl'] )
      print("Training:: sum(r_l)= {}".format(sum(r_l) ) )
      print("Training:: freq of correct= {}".format(sum(self.action_correctness_l)/len(self.action_correctness_l) ) )
      self.scher.train(s_l, a_l, r_l)
      self.jid_info_m.clear()

def learning_shortestq_wsingletraj():
  env = simpy.Environment()
  jg = JG(env, ar=2.5, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1) )
  mq = LearningShortestQ_wSingleTraj(env, n=3)
  jg.out = mq
  jg.init()
  env.run(until=1000*100) # 50000*1

# ***************  Learning shortest-q scheduling with multiple trajectories  ******************** #
# High variance with single trajectory is resolved with multiple trajectory sampling
class LearningShortestQ_wMultTrajs(object):
  def __init__(self, env, n, scher, max_numj, sching=None, act_max=False):
    self.env = env
    self.n = n
    self.scher = scher
    self.max_numj = max_numj
    self.sching = sching
    self.act_max = act_max
    
    slowdown_dist = Dolly() # DUniform(1, 1) # Bern(1, 10, 0.1)
    self.q_l = [FCFS(i, env, slowdown_dist, out_c=self) for i in range(self.n) ]
    self.jid_info_m = {}
    self.num_jcompleted = 0
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
  
  def __repr__(self):
    return "LearningShortestQ_wMultTrajs[n= {}]".format(self.n)
  
  def state(self):
    # s = np.array([q.length() for q in self.q_l] )
    # sum_s = sum(s)
    # return s/sum_s if sum_s != 0 else s
    return [q.length() for q in self.q_l]
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      if self.sching == "jshortestq":
        a = np.argmin(s)
      else:
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
      
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.tsize, 's': s, 'a': a}
      self.q_l[a].put(Task(j._id, j.k, j.tsize, j.tsize) )
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    jinfo = self.jid_info_m[jid]
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    
    self.num_jcompleted += 1
    if self.num_jcompleted > self.max_numj:
      self.env.exit()
  
def learning_shortestq_wmulttrajs():
  num_server = 3
  s_len, a_len, nn_len = num_server, num_server, 10
  straj_training = False # True
  scher = DeepScher(s_len, a_len, nn_len, straj_training)
  
  N, T = 10, 1000 # 10, 100
  
  def sample_traj(T, sching=None, act_max=False):
    reward = lambda sl : 1/sl
    # reward = lambda sl : 101 - sl
    
    env = simpy.Environment()
    jg = JG(env, ar=0.5, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
    mq = LearningShortestQ_wMultTrajs(env, num_server, scher, T, sching, act_max)
    jg.out = mq
    jg.init()
    env.run(until=50000)
    
    t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
    for t in range(T):
      jinfo_m = mq.jid_info_m[t+1]
      t_s_l[t, :] = jinfo_m['s']
      t_a_l[t, :] = jinfo_m['a']
      t_r_l[t, :] = reward(jinfo_m['sl'] )
      t_sl_l[t, :] = jinfo_m['sl']
    # print("t_r_l= {}".format(t_r_l) )
    return t_s_l, t_a_l, t_r_l, t_sl_l
  
  def evaluate(T, sching=None):
    num_shortest_found = 0
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(T, sching, act_max=False)
    for t in range(T):
      s, a = t_s_l[t], int(t_a_l[t][0] )
      # print("s= {}, a= {}".format(s, a) )
      if s[a] - s.min() < 0.1:
        num_shortest_found += 1
    print("avg sl= {}, freq shortest found= {}".format(np.mean(t_sl_l), num_shortest_found/T) )
  # 
  if straj_training:
    print("Eval with jshortestq:")
    for _ in range(5):
      evaluate(T, sching='jshortestq')
    # value_ester = ValueEster(s_len)
    for i in range(100*100):
      t_s_l, t_a_l, t_r_l, _ = sample_traj(T)
      # value_ester.train_w_single_traj(t_s_l, t_r_l)
      scher.train_w_single_traj(t_s_l, t_a_l, t_r_l)
      
      if i % 10 == 0:
        print("i= {}".format(i) )
        evaluate(T)
  else:
    print("Eval before training:")
    for _ in range(3):
      evaluate(T)
    print("Eval with jshortestq:")
    for _ in range(3):
      evaluate(T, sching='jshortestq')
    for i in range(100*100):
      n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(T)
        n_t_s_l[n, :] = t_s_l
        n_t_a_l[n, :] = t_a_l
        n_t_r_l[n, :] = t_r_l
        n_t_sl_l[n, :] = t_sl_l
      num_shortest_found = 0
      for n in range(N):
        for t in range(T):
          s, a = n_t_s_l[n, t], int(n_t_a_l[n, t][0] )
          if s[a] - s.min() < 0.01:
            num_shortest_found += 1
      print("i= {}, avg sl= {}, freq shortest found= {}".format(i, np.mean(n_t_sl_l), num_shortest_found/N/T) )
      scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      # if i % 5 == 0:
      #   evaluate(T)
    print("Eval after learning:")
    evaluate(40000)

if __name__ == "__main__":
  # learning_shortestq_wsingletraj()
  learning_shortestq_wmulttrajs()
