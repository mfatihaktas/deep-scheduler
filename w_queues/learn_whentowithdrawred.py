import numpy as np

from patch import *
from rvs import *
from sim import *
from scher import PolicyGradScher

class MultiQ_Reptod(object):
  def __init__(self, env, n, max_numj, sching_m, scher=None, act_max=False):
    self.env = env
    self.n = n
    self.max_numj = max_numj
    self.sching_m = sching_m
    self.scher = scher
    self.act_max = act_max
    
    self.d = self.sching_m['d']
    
    self.jq = JQ(env, list(range(self.n) ), self)
    
    L = sching_m['L'] if 'L' in sching_m else None
    slowdown_dist = Exp(1) # Exp(1, D=1) # Dolly() # TPareto(1, 200, 1) # DUniform(1, 1)
    self.q_l = [FCFS(i, env, slowdown_dist, out=self.jq, out_c=self.jq, L=L) for i in range(self.n) ]
    
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.num_jcompleted = 0
    self.jsl_l = []
  
  def __repr__(self):
    return "MultiQ_Reptod[n= {}]".format(self.n)
  
  def get_rand_d(self):
    qi_l = random.sample(range(self.n), self.d)
    ql_i_l = sorted([(self.q_l[i].length(), i) for i in qi_l] )
    qi_l = [ql_i[1] for ql_i in ql_i_l]
    ql_l = [ql_i[0] for ql_i in ql_i_l]
    # ql_l = [self.q_l[i].length() for i in qi_l]
    return qi_l, ql_l
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      qi_l, ql_l = self.get_rand_d()
      s = ql_l
      if 'reptod' in self.sching_m:
        toi_l = qi_l
      elif 'reptod-ifidle' in self.sching_m:
        i = 0
        while i < len(ql_l) and ql_l[i] == 0: i += 1
        toi_l = qi_l[:i+1]
      elif 'reptod-withdraw' in self.sching_m:
        toi_l = qi_l
      elif 'reptod-withdraw-wlearning' in self.sching_m:
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
        toi_l = qi_l
      if 'a' not in locals():
        a = 0
      
      if 'reptod-withdraw' in self.sching_m:
        for _, i in enumerate(toi_l):
          if _ == 0:
            self.q_l[i].put(Task(j._id, j.k, j.tsize) )
          else:
            self.q_l[i].put(Task(j._id, j.k, j.tsize, type_='r') )
      elif 'reptod-withdraw-wlearning' in self.sching_m:
        for _, i in enumerate(toi_l):
          if _ == 0:
            self.q_l[i].put(Task(j._id, j.k, j.tsize) )
          else:
            self.q_l[i].put(Task(j._id, j.k, j.tsize, type_='r', L=a) )
      else:
        for i in toi_l:
          self.q_l[i].put(Task(j._id, j.k, j.tsize) )
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.tsize, 'qid_l': toi_l, 's': s, 'a': a}
  
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
    self.num_jcompleted += 1
    if self.num_jcompleted > self.max_numj:
      self.env.exit()

def sample_traj(ns, ar, T, sching_m, scher):
  reward = lambda sl : 100/sl
  
  env = simpy.Environment()
  jg = JG(env, ar, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
  mq = MultiQ_Reptod(env, ns, T, sching_m, scher)
  jg.out = mq
  jg.init()
  env.run()
  
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, sching_m['d'])), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  for t in range(T):
    jinfo_m = mq.jid_info_m[t+1]
    t_s_l[t, :] = jinfo_m['s']
    t_a_l[t, :] = jinfo_m['a']
    t_r_l[t, :] = reward(jinfo_m['sl'] )
    t_sl_l[t, :] = jinfo_m['sl']
  return t_s_l, t_a_l, t_r_l, t_sl_l

def evaluate(ns, ar, T, sching_m, scher):
  _, t_a_l, t_r_l, t_sl_l = sample_traj(ns, ar, T, sching_m, scher)
  print("avg a= {}, avg sl= {}".format(np.mean(t_a_l), np.mean(t_sl_l) ) )

def learn_whentowithdraw():
  ns, d, ar = 10, 3, 2 # 10
  print("ns= {}, d= {}, ar= {}".format(ns, d, ar) )
  s_len, a_len = d, 5
  nn_len = 10
  scher = PolicyGradScher(s_len, a_len, nn_len)
  
  N, T = 10, 1000
  
  print("BEFORE training")
  sching_m = {'reptod-ifidle': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)
  
  sching_m = {'reptod-withdraw': 0, 'd': d, 'L': 0}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)
  
  sching_m = {'reptod-withdraw-wlearning': 0, 'd': d}
  for i in range(25):
    print("i= {}".format(i) )
    n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
    for n in range(N):
      s_l, a_l, r_l, sl_l = sample_traj(ns, ar, T, sching_m, scher)
      n_t_s_l[n, :] = s_l
      n_t_a_l[n, :] = a_l
      n_t_r_l[n, :] = r_l
      n_t_sl_l[n, :] = sl_l
    print("avg a= {}, training avg sl= {}".format(np.mean(n_t_a_l), np.mean(n_t_sl_l) ) )
    scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
    # if i % 1 == 0:
    #   print("eval:")
    #   evaluate(ns, ar, T, sching_m, scher)
    
    if i % 5:
      continue
    for j in range(20):
      for _ in range(3):
        s = np.random.randint(j*5+1, size=s_len)
        a = scher.get_max_action(s) # scher.get_random_action(s)
        print("s= {}, a= {}".format(s, a) )
  print("AFTER training")
  T_ = 20*T
  sching_m = {'reptod-withdraw-wlearning': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T_, sching_m, scher)
  
  sching_m = {'reptod-ifidle': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)
  
  sching_m = {'reptod-withdraw': 0, 'd': d, 'L': 0}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)

if __name__ == "__main__":
  learn_whentowithdraw()