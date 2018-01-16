import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scheduling import DeepScher

class LearningRepSching(object):
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
    return "LearningRepSching[n= {}]".format(self.n)
  
  def state(self):
    # s = np.array([q.length() for q in self.q_l] )
    # sum_s = sum(s)
    # return s/sum_s if sum_s != 0 else s
    return [q.length() for q in self.q_l]
  
  def get_sorted_qids(self):
    # qid_length_m = {q._id: q.length() for q in self.q_l}
    # qid_length_l = sorted(qid_length_m.items(), key=operator.itemgetter(1) )
    # return [qid_length[0] for qid_length in qid_length_l]
    t_l = sorted([(q.length(), q._id) for q in self.q_l] )
    return [t[1] for t in t_l]
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      s = self.state()
      if self.sching_m is None:
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
      else:
        a = self.sching_m['n'] - 1
      toi_l = random.sample(range(self.n), a+1)
      # toi_l = self.get_sorted_qids()[0:a+1]
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

def learning_repsching_w_mult_trajs():
  num_server = 3
  s_len, a_len = num_server, num_server
  N, T = 2, 10000
  scher = DeepScher(s_len, a_len)
  
  def sample_traj(T, act_max=False, sching_m=None):
    reward = lambda sl : 1/sl
    # reward = lambda sl : 1 - sl
    
    env = simpy.Environment()
    jg = JG(env, ar=0.5, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
    mq = LearningRepSching(env, num_server, scher, T, act_max, sching_m)
    jg.out = mq
    jg.init()
    env.run() # until=50000
    
    s_l, a_l, r_l, sl_l = [], [], [], []
    for jid in range(1, T+1):
      jinfo_m = mq.jid_info_m[jid]
      # print("jid= {}, jinfo_m= \n{}".format(jid, pprint.pformat(jinfo_m) ) )
      
      s_l.append(jinfo_m['s'] )
      a_l.append(jinfo_m['a'] )
      r_l.append(reward(jinfo_m['sl'] ) )
      sl_l.append(jinfo_m['sl'] )
    # min_sl = min(sl_l)
    # r_l = [min_sl/mq.jid_info_m[jid]['sl'] for jid in range(1, T+1) ]
    # print("r_l= {}".format(r_l) )
    return s_l, a_l, r_l, sl_l
  
  def evaluate(T, sching_m=None):
    _, a_l, r_l, sl_l = sample_traj(T, True, sching_m)
    # print("a_l= {}".format(a_l) )
    print("avg a= {}".format(np.mean(a_l) ) )
    print("avg slowdown= {}".format(np.mean(sl_l) ) )
  # 
  sching_m = {'n': num_server}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(2):
    evaluate(T, sching_m)
  
  sching_m = {'n': 1}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(2):
    evaluate(T, sching_m)
  
  # print("Eval before training:")
  # for _ in range(3):
  #   evaluate(T)
  for i in range(100):
    n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((N, T, s_len)), np.zeros((N, T)), np.zeros((N, T))
    for n in range(N):
      s_l, a_l, r_l, _ = sample_traj(T)
      n_t_s_l[n, :] = s_l
      n_t_a_l[n, :] = a_l
      n_t_r_l[n, :] = r_l
    scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
    if i % 1 == 0:
      print("i= {}".format(i) )
      evaluate(T)
  print("Eval after learning:")
  evaluate(T*1)
  
  for i in range(100):
    if i == 0:
      s = [0]*num_server
    else:
      s = np.random.randint(100, size=num_server)
    a = scher.get_max_action(s)
    print("s= {}, a= {}".format(s, a) )

if __name__ == "__main__":
  learning_repsching_w_mult_trajs()
