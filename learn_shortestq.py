import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scher import *

# ***************  Learning shortest-q scheduling with multiple trajectories  ******************** #
class MultiQ(object):
  def __init__(self, env, ns, max_numj, scher, sching=None, act_max=False):
    self.env = env
    self.ns = ns
    self.max_numj = max_numj
    self.scher = scher
    self.sching = sching
    self.act_max = act_max
    
    slowdown_dist = Dolly() # DUniform(1, 1) # Bern(1, 10, 0.1)
    self.q_l = [FCFS(i, env, slowdown_dist, out_c=self) for i in range(self.ns) ]
    self.jid_info_m = {}
    self.num_jcompleted = 0
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
  
  def __repr__(self):
    return "MultiQ[ns= {}]".format(self.ns)
  
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

def sample_traj(scher, ns, ar, T, sching=None, act_max=False):
  reward = lambda sl : 1/sl
  
  env = simpy.Environment()
  jg = JG(env, ar, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
  mq = MultiQ(env, ns, T, scher, sching, act_max)
  jg.out = mq
  jg.init()
  env.run()
  
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, ns)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  for t in range(T):
    jinfo_m = mq.jid_info_m[t+1]
    t_s_l[t, :] = jinfo_m['s']
    t_a_l[t, :] = jinfo_m['a']
    t_r_l[t, :] = reward(jinfo_m['sl'] )
    t_sl_l[t, :] = jinfo_m['sl']
  return t_s_l, t_a_l, t_r_l, t_sl_l
  # return np.random.rand(T, s_len), np.random.rand(T, 1), np.random.rand(T, 1), np.random.rand(T, 1)

def evaluate(scher, ns, ar, T, sching=None):
  num_shortest_found = 0
  t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(scher, ns, ar, T, sching, act_max=True)
  for t in range(T):
    s, a = t_s_l[t], int(t_a_l[t][0] )
    if s[a] - s.min() < 0.1:
      num_shortest_found += 1
  print("avg sl= {}, freq shortest found= {}".format(np.mean(t_sl_l), num_shortest_found/T) )

def learn_shortestq():
  ns = 3
  ar = 0.5
  s_len, a_len, nn_len = ns, ns, 10
  scher = PolicyGradScher(s_len, a_len, nn_len, straj_training=False)
  
  N, T = 10, 1000
  
  print("Eval before training:")
  for _ in range(3):
    evaluate(scher, ns, ar, T)
  print("Eval with jshortestq:")
  for _ in range(3):
    evaluate(scher, ns, ar, T, sching='jshortestq')
  for i in range(100*100):
    n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
    for n in range(N):
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(scher, ns, ar, T)
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
    if i > 0:
      scher.restore(i-1)
    scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
    scher.save(i)
    if i % 5 == 0:
      evaluate(scher, ns, ar, 4*T)
  print("Eval after learning:")
  evaluate(scher, ns, ar, T=40000)

if __name__ == "__main__":
  learn_shortestq()
