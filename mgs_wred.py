import math, random, simpy, pprint
import numpy as np

from sim import Task, JG
from patch import *
from mgs_wred_model import *

# ###########################################  Sim  ############################################## #
class Server(object):
  def __init__(self, _id, env, S, out_c):
    self._id = _id
    self.env = env
    self.S = S
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
  
  def length(self):
    return len(self.store.items) + (self.t_inserv != None)
  
  def run(self):
    while 1:
      self.t_inserv = (yield self.store.get() )
      self.cancel = self.env.event()
      clk_start_time = self.env.now
      st = self.t_inserv.size * (self.S.gen_sample() + random.randint(1, 10)/1000)
      sim_log(DEBUG, self.env, self, "starting {}s-clock on ".format(st), self.t_inserv)
      yield (self.cancel | self.env.timeout(st) )
      if self.cancel_flag:
        sim_log(DEBUG, self.env, self, "cancelled clock on ", self.t_inserv)
        self.cancel_flag = False
      else:
        sim_log(DEBUG, self.env, self, "serv done in {}s on ".format(self.env.now-clk_start_time), self.t_inserv)
        self.out_c.put_c({'qid': self._id, 'jid': self.t_inserv.jid} )
      self.t_inserv = None
      # if self.length() > 1:
      #   sim_log(ERROR, self.env, self, "length= {}".format(self.length() ), self.t_inserv)
      #   # print("store= ".format(["{}".format(t) for t in self.store.items] ) )
      #   # print("len(store.items)= {}, store= ".format(len(self.store.items), self.store) )
      #   nt = (yield self.store.get() )
      #   print("nt= {}, len(store.items)= {}".format(nt, len(self.store.items) ) )
  
  def put(self, t):
    sim_log(DEBUG, self.env, self, "recved", t)
    self.store.put(t)
    if self.t_inserv is not None and self.t_inserv.type_ == 'r':
      self.cancel_flag = True
      self.cancel.succeed()
      # self.t_inserv = None
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    if self.t_inserv is not None and jid == self.t_inserv.jid:
      self.cancel_flag = True
      self.cancel.succeed()
      # self.t_inserv = None

class MGn(object):
  def __init__(self, env, n, sching, S, max_numj):
    self.env = env
    self.n = n
    self.sching = sching
    self.max_numj = max_numj
    
    self.s_l = [Server(i, env, S, out_c=self) for i in range(self.n) ]
    # self.sbusyf_l = [0]*n
    self.wait_for_frees = None
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.jid_info_m = {}
    
    self.num_jcompleted = 0
  
  def __repr__(self):
    return "MGn[n= {}]".format(self.n)
  
  def length(self):
    # return sum([1 for f in self.sbusyf_l if f > 0] ) + len(self.store.items)
    return sum([1 for s in self.s_l if s.length() != 0 and s.t_inserv.type_ != 'r'] ) \
           + len(self.store.items)
  
  def free_sid(self):
    if self.sching == 'no-rep':
      # i = 0
      # while i < self.n and self.sbusyf_l[i] == 1: i += 1
      # return [i] if i < self.n else []
      return [i for i, s in enumerate(self.s_l) if s.length() == 0]
    elif self.sching == 'rep':
      # l0 = [i for i, f in enumerate(self.sbusyf_l) if f == 0]
      # l2 = [i for i, f in enumerate(self.sbusyf_l) if f == 2]
      l0 = [i for i, s in enumerate(self.s_l) if s.length() == 0]
      lr = [i for i, s in enumerate(self.s_l) if s.length() != 0 and s.t_inserv.type_ == 'r']
      if len(l0) != 0:
        return l0
      elif len(lr) != 0:
        return [lr[0]]
      else:
        return []
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      toi_l = self.free_sid()
      if len(toi_l) == 0:
        self.wait_for_frees = self.env.event()
        sim_log(DEBUG, self.env, self, "wait_for_frees", j)
        yield (self.wait_for_frees)
        self.wait_for_frees = None
        toi_l = self.free_sid()
      
      # print("toi_l= {}".format(toi_l) )
      self.jid_info_m[j._id]['sid_l'] = toi_l
      # self.sbusyf_l[toi_l[0] ] = 1
      self.s_l[toi_l[0] ].put(Task(j._id, j.k, j.tsize) )
      if self.sching == 'rep' and len(self.store.items) == 0:
        for i in toi_l[1:]:
          # self.sbusyf_l[i] = 2
          self.s_l[i].put(Task(j._id, j.k, j.tsize, 'r') )
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    # print("sbusyf_l= {}".format(self.sbusyf_l) )
    self.jid_info_m[j._id] = {'ent': self.env.now, 'fs': self.length() }
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid, qid = m['jid'], m['qid']
    self.jid_info_m[jid]['t'] = self.env.now - self.jid_info_m[jid]['ent']
    # self.sbusyf_l[qid] = 0
    if self.sching == 'rep':
      sid_l = self.jid_info_m[jid]['sid_l']
      for i in sid_l:
        if i != qid:
          # self.sbusyf_l[i] = 0
          self.s_l[i].put_c({'m': 'cancel', 'jid': jid} )
    
    self.num_jcompleted += 1
    if self.num_jcompleted > self.max_numj:
      log(DEBUG, "num_jcompleted= {} > max_numj= {}".format(self.num_jcompleted, self.max_numj) )
      self.env.exit()
    
    if self.wait_for_frees is not None:
      # print("freed up res; m= {}".format(m) )
      self.wait_for_frees.succeed()
  
def sim_mgn(nf, ar, n, sching, ts_dist, S, max_numj):
  ET = 0
  for _ in range(nf):
    env = simpy.Environment()
    jg = JG(env, ar, DUniform(1, 1), ts_dist, max_numj)
    q = MGn(env, n, sching, S, max_numj)
    jg.out = q
    jg.init()
    env.run()
    
    # print("q.jid_info_m=\n{}".format(pprint.pformat(q.jid_info_m) ) )
    ET += np.mean([q.jid_info_m[j+1]['t'] for j in range(max_numj) ] )
    
    fs_ntimes_m = {}
    for j in range(max_numj):
      s = q.jid_info_m[j+1]['fs']
      if s not in fs_ntimes_m:
        fs_ntimes_m[s] = 0
      fs_ntimes_m[s] += 1
    fs_freq_m = {s: ntimes/max_numj for s, ntimes in fs_ntimes_m.items() }
    # print("fs_freq_m=\n {}".format(pprint.pformat(fs_freq_m) ) )
    print("fs_freq_m[0]= {}".format(fs_freq_m[0] ) )
    '''
    mu = S.mu
    err = 0
    for s, freq in fs_freq_m.items():
      err += abs(freq - mmn_prob_jinsys(n, ar, mu, s) )
    print("err= {}".format(err) )
    '''
  return ET/nf

def plot_mgn():
  n = 4
  ts_dist = DUniform(1, 1)
  S = Exp(1) # Pareto(1, 5)
  mu = 1 # 1/moment_ith(S, 1)
  
  V = S
  max_numj = 30000
  log(WARNING, "n= {}, ts_dist= {}, S= {}".format(n, ts_dist, S) )
  
  nf = 1
  for ar in np.linspace(0.05, 0.9*n*mu, 5):
  # for ar in np.linspace(1, 0.9*n*mu, 1):
    print("\nar= {}".format(ar) )
    # sching = 'no-rep'
    # print("sching= {}".format(sching) )
    # ET = sim_mgn(nf, ar, n, sching, ts_dist, S, max_numj)
    # print("ET= {}".format(ET) )
    # ETm = mmn_ET(n, ar, mu) # mgn_ET(n, ar, V)
    # print("ETm= {}".format(ETm) )
    
    sching = 'rep'
    print("sching= {}".format(sching) )
    # ET = sim_mgn(nf, ar, n, sching, ts_dist, S, max_numj)
    # print("ET= {}".format(ET) )
    ETm = mgn_rep_ET(n, ar, V)
    print("ETm= {}".format(ETm) )
  

if __name__ == "__main__":
  plot_mgn()
