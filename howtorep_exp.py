import numpy as np

from patch import *
from rvs import *
from sim import *

class MultiQ_wRep(object):
  def __init__(self, env, n, max_numj, sching_m, scher=None, act_max=False):
    self.env = env
    self.n = n
    self.max_numj = max_numj
    self.sching_m = sching_m
    self.scher = scher
    self.act_max = act_max
    
    self.jq = JQ(env, list(range(self.n) ), self)
    slowdown_dist = Dolly() # DUniform(1, 1)
    
    L = sching_m['L'] if 'L' in sching_m else None
    self.q_l = [FCFS(i, env, slowdown_dist, out=self.jq, L=L) for i in range(self.n) ]
    
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.num_jcompleted = 0
    self.jsl_l = []
  
  def __repr__(self):
    return "MultiQ_wRep[n= {}]".format(self.n)
  
  def get_rand_d(self):
    # return np.array([q.length() for q in self.q_l] )
    i_l = random.sample(range(self.n), self.sching_m['d'] )
    ql_i_l = sorted([(self.q_l[i].length(), i) for i in i_l] )
    
    i_l = [ql_i[1] for ql_i in ql_i_l]
    ql_l = [ql_i[0] for ql_i in ql_i_l]
    return i_l, ql_l
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      i_l, ql_l = self.get_rand_d()
      s = ql_l
      if 'rep-to-d' in self.sching_m:
        toi_l = i_l
      elif 'rep-to-d-ifidle' in self.sching_m:
        i = 0
        while i < len(ql_l) and ql_l[i] == 0: i += 1
        toi_l = i_l[:i+1]
      elif 'rep-to-d-wcancel' in self.sching_m:
        toi_l = i_l
      elif 'rep-to-d-wlearning' in self.sching_m:
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
        toi_l = i_l[:a+1]
      
      a = len(toi_l) - 1
      if 'rep-to-d-wcancel' in self.sching_m:
        for _, i in enumerate(toi_l):
          if _ == 0:
            self.q_l[i].put(Task(j._id, j.k, j.tsize) )
          else:
            self.q_l[i].put(Task(j._id, j.k, j.tsize, type_='r') )
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

def avg_sl(nf, ar, ns, T, sching_m):
  sl = 0
  for _ in range(nf):
    env = simpy.Environment()
    jg = JG(env, ar, k_dist=DUniform(1, 1), tsize_dist=DUniform(1, 1), max_sent=T)
    mq = MultiQ_wRep(env, ns, T, sching_m)
    jg.out = mq
    jg.init()
    env.run()
    
    sl += np.mean([mq.jid_info_m[t+1]['sl'] for t in range(T) ] )
  return sl/nf

def plot_reptod_ifidle_vs_wcancel():
  ns, d = 10, 3
  T = 30000
  alog("ns= {}, d= {}, T= {}".format(ns, d, T) )
  
  ar_l = []
  sl_ifidle_l, sl_wcancel_l = [], []
  nf = 1
  # for ar in np.linspace(0.1, 2, 7):
  for ar in np.linspace(1.9, 2.1, 2):
    print("\n ar= {}".format(ar) )
    ar_l.append(ar)
    
    sching_m = {'rep-to-d-ifidle': 0, 'd': d}
    sl = avg_sl(nf, ar, ns, T, sching_m)
    print("sching_m= {} \n sl= {}".format(sching_m, sl) )
    sl_ifidle_l.append(sl)
    
    sching_m = {'rep-to-d-wcancel': 0, 'd': d, 'L': 4}
    sl = avg_sl(nf, ar, ns, T, sching_m)
    print("sching_m= {} \n sl= {}".format(sching_m, sl) )
    sl_wcancel_l.append(sl)
  plot.plot(ar_l, sl_ifidle_l, label='rep-to-d-ifidle', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.plot(ar_l, sl_wcancel_l, label='rep-to-d-wcancel', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  
  plot.title(r'$n= {}$, $d= {}$'.format(ns, d) )
  plot.xlabel(r'$\lambda$', fontsize=14)
  plot.ylabel(r'Avg Slowdown', fontsize=14)
  plot.legend()
  plot.savefig("plot_reptod_ifidle_vs_wcancel.pdf")
  log(WARNING, "done.")
  
if __name__ == "__main__":
  plot_reptod_ifidle_vs_wcancel()
