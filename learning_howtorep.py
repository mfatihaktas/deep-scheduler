import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scheduling import DeepScher

class LearningHowtoRep(object):
  def __init__(self, env, n, scher, max_numj, act_max=False, sching_m=None):
    self.env = env
    self.n = n
    self.scher = scher
    self.max_numj = max_numj
    self.act_max = act_max
    self.sching_m = sching_m
    
    self.jq = JQ(env, list(range(self.n) ), self)
    # self.q_l = [PSQ(i, env, h=4, out=self.jq) for i in range(self.n) ]
    slowdown_dist = Dolly() # DUniform(1, 1)
    self.q_l = [FCFS(i, env, slowdown_dist, out=self.jq) for i in range(self.n) ]
    
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.num_jcompleted = 0
    self.jsl_l = []
  
  def __repr__(self):
    return "LearningHowtoRep[n= {}]".format(self.n)
  
  def state(self):
    s = np.array([q.length() for q in self.q_l] )
    # sum_s = sum(s)
    # return s/sum_s if sum_s != 0 else s
    return s
  
  def get_sorted_qids(self):
    # qid_length_m = {q._id: q.length() for q in self.q_l}
    # qid_length_l = sorted(qid_length_m.items(), key=operator.itemgetter(1) )
    # return [qid_length[0] for qid_length in qid_length_l]
    t_l = sorted([(q.length(), q._id) for q in self.q_l] )
    return [t[1] for t in t_l]
  
  def sample_qids(self, n):
    # return random.sample(range(self.n), n)
    return self.get_sorted_qids()[0:n]
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      if self.sching_m is None:
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
        toi_l = self.sample_qids(a+1)
      elif 'rep-to-idle' in self.sching_m:
        toi_l = [i for i, ql in enumerate(s) if ql == 0]
        if len(toi_l) == 0:
          a = 0
          toi_l = self.sample_qids(a+1)
        else:
          a = len(toi_l) - 1
      else:
        a = self.sching_m['n'] - 1
        toi_l = self.sample_qids(a+1)
      
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

def learning_howtorep_w_mult_trajs():
  num_server = 6
  s_len, a_len = num_server, num_server
  nn_len = 10
  straj_training = False
  scher = DeepScher(s_len, a_len, nn_len, straj_training)
  
  N, T = 10, 1000
  
  def sample_traj(T, act_max=False, sching_m=None):
    # reward = lambda sl : 1/sl
    def reward(sl):
      # if sl < 12:
      #   sl = 1
      return 100/sl # 101 - sl
    
    env = simpy.Environment()
    # ar=0.3, n=3
    # ar=0.6, n=6
    jg = JG(env, ar=0.6, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
    mq = LearningHowtoRep(env, num_server, scher, T, act_max, sching_m)
    jg.out = mq
    jg.init()
    env.run() # until=50000
    
    t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
    for t in range(T):
      jinfo_m = mq.jid_info_m[t+1]
      # print("t= {}, jinfo_m= {}".format(t, jinfo_m) )
      t_s_l[t, :] = jinfo_m['s']
      t_a_l[t, :] = jinfo_m['a']
      t_r_l[t, :] = reward(jinfo_m['sl'] )
      t_sl_l[t, :] = jinfo_m['sl']
    return t_s_l, t_a_l, t_r_l, t_sl_l
    
  def evaluate(T, sching_m=None):
    _, t_a_l, t_r_l, t_sl_l = sample_traj(T, False, sching_m)
    # print("t_a_l= {}".format(t_a_l) )
    print("avg a= {}, avg sl= {}".format(np.mean(t_a_l), np.mean(t_sl_l) ) )
  # 
  print("BEFORE training")
  sching_m = {'rep-to-idle': 0}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(T, sching_m)
  
  for n in range(1, num_server+1):
    sching_m = {'n': n}
    print("Eval with sching_m= {}".format(sching_m) )
    for _ in range(3):
      evaluate(T, sching_m)
  # print("Eval before training:")
  # for _ in range(3):
  #   evaluate(T)
  for i in range(100):
    print("i= {}".format(i) )
    n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
    for n in range(N):
      s_l, a_l, r_l, sl_l = sample_traj(T)
      n_t_s_l[n, :] = s_l
      n_t_a_l[n, :] = a_l
      n_t_r_l[n, :] = r_l
      n_t_sl_l[n, :] = sl_l
    print("avg a= {}, training avg sl= {}".format(np.mean(n_t_a_l), np.mean(n_t_sl_l) ) )
    scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
    # if i % 1 == 0:
    #   print("eval:")
    #   evaluate(T)
    if i % 5:
      continue
    for j in range(20):
      for _ in range(3):
        if j == 0:
          s = [0]*num_server
        else:
          s = np.random.randint(j*5, size=num_server)
          # sum_s = sum(s)
          # s = s if sum_s == 0 else s/sum_s
        a = scher.get_random_action(s)
        print("s= {}, a= {}".format(s, a) )
  print("AFTER training")
  evaluate(T=40000)
  
  sching_m = {'rep-to-idle': 0}
  print("Eval with sching_m= {}".format(sching_m) )
  evaluate(T, sching_m)
  
  for n in range(1, num_server+1):
    sching_m = {'n': n}
    print("Eval with sching_m= {}".format(sching_m) )
    evaluate(T, sching_m)
  
if __name__ == "__main__":
  learning_howtorep_w_mult_trajs()
