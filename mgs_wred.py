import math, random, simpy, pprint
import numpy as np

from sim import Task, JG
from patch import *
from rvs import *

def prob_jinsys_mmn(ar, mu, n, j):
  ro = ar/mu/n
  
  P0 = 1/sum([ro**i/fact(i) + ro**n/fact(n)*(n*mu/(n*mu - ar) ) for i in range(n) ] )
  if j <= n:
    return ro**j/fact(j) * P0
  else:
    return ro**j/fact(n)/(n)**(j-n) * P0


# ###########################################  Sim  ############################################## #
class Server(object):
  def __init__(self, _id, env, slowdown_dist, out_c):
    self._id = _id
    self.env = env
    self.slowdown_dist = slowdown_dist
    self.out_c = out_c
    
    self.t_inserv = None
    self.cancel_flag = False
    self.cancel = None
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
  def __repr__(self):
    return "Server[_id= {}]".format(self._id)
  
  def busy(self):
    return self.t_inserv == None
  
  def run(self):
    self.t_inserv = (yield self.store.get() )
    self.cancel = self.env.event()
    # clk_start_time = self.env.now
    st = self.t_inserv.size * self.slowdown_dist.gen_sample()
    # sim_log(DEBUG, self.env, self, "starting {}s-clock on ".format(st), self.t_inserv)
    yield (self.cancel | self.env.timeout(st) )
    if self.cancel_flag:
      # sim_log(DEBUG, self.env, self, "cancelled clock on ", self.t_inserv)
      self.cancel_flag = False
    else:
      # sim_log(DEBUG, self.env, self, "serv done in {}s on ".format(self.env.now-clk_start_time), self.t_inserv)
      self.out_c.put_c({'qid': self._id, 'jid': self.t_inserv.jid} )
    self.t_inserv = None
  
  def put(self, t):
    sim_log(DEBUG, self.env, self, "recved", t)
    self.store.put(t)
    if self.t_inserv is not None and self.t_inserv.type_ == 'r':
      self.cancel_flag = True
      self.cancel.succeed()
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    # if m['m'] == 'cancel':
    jid = m['jid']
    if jid == self.t_inserv.jid:
      self.cancel_flag = True
      self.cancel.succeed()

class MGn(object):
  def __init__(self, env, n, sching, sl_dist, max_numj):
    self.env = env
    self.n = n
    self.sching = sching
    self.max_numj = max_numj
    
    self.s_l = [Server(i, env, sl_dist, out_c=self) for i in range(self.n) ]
    self.sbusyf_l = [0]*n
    self.wait_for_frees = None
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.jid_info_m = {}
    
    self.num_jcompleted = 0
  
  def __repr__(self):
    return "MGn[n= {}]".format(self.n)
  
  def length(self):
    return len(self.free_sid() ) + len(self.store.items)
  
  def free_sid(self):
    if self.sching == 'no-rep':
      i = 0
      while i < self.n and self.sbusyf_l[i] > 0: i += 1
      return [i] if i < self.n else []
    elif self.sching == 'rep':
      return [i for i, f in enumerate(self.sbusyf_l) if f == 0]
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      toi_l = self.free_sid()
      if len(toi_l) == 0:
        self.wait_for_frees = self.env.event()
        yield (self.wait_for_frees)
        self.wait_for_frees = None
        toi_l = self.free_sid()
      
      self.jid_info_m[j._id]['sid_l'] = toi_l
      self.sbusyf_l[toi_l[0] ] = 1
      self.s_l[toi_l[0] ].put(Task(j._id, j.k, j.tsize) )
      if self.sching == 'rep':
        for i in toi_l[1:]:
          # self.sbusyf_l[i] = 1
          self.s_l[i].put(Task(j._id, j.k, j.tsize, 'r') )
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    self.jid_info_m[j._id] = {'ent': self.env.now, 'foundstate': self.length() }
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid, qid = m['jid'], m['qid']
    self.jid_info_m[jid]['t'] = self.env.now - self.jid_info_m[jid]['ent']
    self.sbusyf_l[qid] = 0
    if self.sching == 'rep':
      sid_l = self.jid_info_m[jid]['sid_l']
      for i in sid_l:
        if i != qid:
          self.s_l[i].put_c({'m': 'cancel', 'jid': jid} )
          self.sbusyf_l[i] = 0
    
    # self.num_jcompleted += 1
    # if self.num_jcompleted > self.max_numj:
    #   log(DEBUG, "num_jcompleted= {} > max_numj= {}".format(self.num_jcompleted, self.max_numj) )
    #   self.env.exit()
    
    if self.wait_for_frees is not None:
      self.wait_for_frees.succeed()
  
def sim_MGn(nf, ar, n, sching, ts_dist, sl_dist, max_numj):
  ET = 0
  for _ in range(nf):
    env = simpy.Environment()
    jg = JG(env, ar, DUniform(1, 1), ts_dist, max_numj)
    q = MGn(env, n, sching, sl_dist, max_numj)
    jg.out = q
    jg.init()
    env.run(until=50000*5)
  print("q.jid_info_m=\n{}".format(pprint.pformat(q.jid_info_m) ) )
  ET += np.mean([q.jid_info_m[j+1]['t'] for j in range(max_numj) ] )
  
  fs_ntimes_m = {}
  for j in range(max_numj):
    s = q.jid_info_m[j+1]['foundstate']
    if s not in fs_ntimes_m:
      fs_ntimes_m[s] = 0
    fs_ntimes_m[s] += 1
  fs_freq_m = {s: ntimes/max_numj for s, ntimes in fs_ntimes_m.items() }
  print("fs_freq_m=\n {}".format(pprint.pformat(fs_freq_m) ) )
  
  return ET/nf

def plot_mgn():
  n = 4
  sching = 'no-rep'
  ts_dist = DUniform(1, 1)
  sl_dist = Exp(1)
  max_numj = 10 # 10000
  log(WARNING, "n= {}, sching= {}, ts_dist= {}, sl_dist= {}".format(n, sching, ts_dist, sl_dist) )
  
  nf = 1
  for ar in np.linspace(0.05, 2, 5):
    ET = sim_MGn(nf, ar, n, sching, ts_dist, sl_dist, max_numj)
    print("ET= {}".format(ET) )

if __name__ == "__main__":
  plot_mgn()
