import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scheduling import DeepScher

class LearningRepSching(object):
  def __init__(self, env, n, scher, act_max=False):
    self.env = env
    self.n = n
    self.scher = scher
    self.act_max = act_max
    
    self.jq = JQ(env, range(self.n) )
    self.jq.out_c = self
    # self.q_l = [PSQ(i, env, h=4, out=self.jq) for i in range(self.n) ]
    slowdown_dist = DUniform(1, 1) # Dolly()
    self.q_l = [FCFS(i, env, slowdown_dist, out=self.jq) for i in range(self.n) ]
    
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.jsl_l = []
  
  def __repr__(self):
    return "LearningRepSching[n= {}]".format(self.n)
  
  def state(self):
    s = np.array([q.length() for q in self.q_l] )
    sum_s = sum(s)
    return s/sum_s if sum_s != 0 else s
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
      a += 1
      
      toi_l = random.sample(range(self.n), a)
      for i in toi_l:
        self.q_l[i].put(Task(j._id, j.k, j.ts, j.ts) )
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.ts, 'qid_l': toi_l, 's': s, 'a': a}
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    jinfo = self.jid_info_m[jid]
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    
    for i in jinfo['qid_l']:
      if i not in m['deped_from']:
        self.q_l[i].put_c({'m': 'cancel', 'jid': jid} )
    
    self.jsl_l.append(t)
    # self.jid_info_m.pop(jid, None)

def learning_repsching_w_mult_trajs():
  num_server = 3
  s_len, a_len = num_server, 2
  N, T = 10, 100
  scher = DeepScher(s_len, a_len)
  
  def sample_traj(T):
    env = simpy.Environment()
    jg = JG(env, ar=2, k_dist=DUniform(1, a_len), tsize_dist=DUniform(1, 1), max_sent=T)
    mq = LearningRepSching(env, num_server, scher)
    jg.out = mq
    jg.init()
    env.run(until=50000)
    
    s_l, a_l, r_l = [], [], []
    for jid in range(1, T+1):
      jinfo_m = mq.jid_info_m[jid]
      
      s_l.append(jinfo_m['s'] )
      a_l.append(jinfo_m['a'] )
      r_l.append(1/jinfo_m['sl'] )
    # print("r_l= {}".format(r_l) )
    return s_l, a_l, r_l
  
  def evaluate(T):
    num_shortest_found = 0
    s_l, a_l, r_l = sample_traj(T)
    for i in range(len(s_l) ):
      s, a = s_l[i], a_l[i]
      if s[a] - min(s) < 0.01:
        num_shortest_found += 1
    print("freq shortest found= {}".format(num_shortest_found/T) )
  # 
  for i in range(100*4):
    n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((N, T, s_len)), np.zeros((N, T)), np.zeros((N, T))
    for n in range(N):
      s_l, a_l, r_l = sample_traj(T)
      n_t_s_l[n, :] = s_l
      n_t_a_l[n, :] = a_l
      n_t_r_l[n, :] = r_l
    scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
    if i % 10 == 0:
      print("i= {}".format(i) )
      evaluate(T)
  print("Final eval")
  evaluate(T*10)
