import numpy as np

from patch import *
from rvs import *
from sim import *

class MultiQ_wRep(object):
  def __init__(self, env, n, max_numj, sching_m, sl_dist, scher=None, act_max=False):
    self.env = env
    self.n = n
    self.max_numj = max_numj
    self.sching_m = sching_m
    self.scher = scher
    self.act_max = act_max
    
    self.d = self.sching_m['d']
    
    self.jq = JQ(env, list(range(self.n) ), self)
    
    L = sching_m['L'] if 'L' in sching_m else None
    self.q_l = [FCFS(i, env, sl_dist, out=self.jq, out_c=self.jq, L=L) for i in range(self.n) ]
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.jid_info_m = {}
    self.num_jcompleted = 0
    self.jsl_l = []
    self.d_numj_l = [0 for _ in range(self.d) ]
  
  def __repr__(self):
    return "MultiQ_wRep[n= {}]".format(self.n)
  
  def get_rand_d(self):
    qi_l = random.sample(range(self.n), self.d)
    return qi_l, None
  
  def get_sortedrand_d(self):
    qi_l = random.sample(range(self.n), self.d)
    ql_i_l = sorted([(self.q_l[i].length(), i) for i in qi_l] )
    qi_l = [ql_i[1] for ql_i in ql_i_l]
    ql_l = [ql_i[0] for ql_i in ql_i_l]
    # ql_l = [self.q_l[i].length() for i in qi_l]
    return qi_l, ql_l
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      
      qi_l, ql_l = self.get_sortedrand_d()
      s = ql_l
      sname = self.sching_m['name']
      if sname == 'reptod':
        toi_l = qi_l
      elif sname == 'reptod-ifidle' or sname == 'reptod-ifidle-wcancel':
        # i = 0
        # while i < len(ql_l) and ql_l[i] == 0: i += 1
        # if i > 0:
        #   toi_l = qi_l[:i]
        # else:
        #   toi_l = [qi_l[random.randint(0, self.d-1) ] ]
        
        qi_l, ql_l = self.get_rand_d()
        toi_l = [qi_l[0] ]
        toi_l += [qi_l[i] for i, l in enumerate(ql_l[1:] ) if l == 0]
      elif 'reptod-wcancel' in self.sching_m:
        qi_l, ql_l = self.get_rand_d()
        
      
      a = (len(toi_l) > 1)
      self.d_numj_l[a] += 1
      if sname == 'reptod-wcancel' or sname == 'reptod-ifidle-wcancel':
        self.q_l[toi_l[0]].put(Task(j._id, j.k, j.size) )
        for i in toi_l[1:]:
          self.q_l[i].put(Task(j._id, j.k, j.size, type_='r') )
      else:
        for i in toi_l:
          self.q_l[i].put(Task(j._id, j.k, j.size) )
      self.jid_info_m[j._id] = {'ent': self.env.now, 'ts': j.size, 'qid_l': toi_l, 's': s, 'a': a}
  
  def put(self, j):
    sim_log(DEBUG, self.env, self, "recved", j)
    return self.store.put(j)
  
  def put_c(self, m):
    sim_log(DEBUG, self.env, self, "recved", m)
    jid = m['jid']
    jinfo = self.jid_info_m[jid]
    self.jid_info_m[jid]['T'] = self.env.now - jinfo['ent']
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    
    for i in jinfo['qid_l']:
      if i not in m['deped_from']:
        self.q_l[i].put_c({'m': 'cancel', 'jid': jid} )
    
    self.jsl_l.append(t)
    self.num_jcompleted += 1
    if self.num_jcompleted > self.max_numj:
      self.env.exit()

def sim_sl(nf, ar, ns, T, sching_m, ts_dist, sl_dist):
  sl, sl2 = 0, 0
  for _ in range(nf):
    env = simpy.Environment()
    jg = JG(env, ar, k_dist=DUniform(1, 1), size_dist=ts_dist, max_sent=T)
    mq = MultiQ_wRep(env, ns, T, sching_m, sl_dist)
    jg.out = mq
    jg.init()
    env.run()
    
    sl += np.mean([mq.jid_info_m[t+1]['sl'] for t in range(T) ] )
    
    # print("jid_r_m= \n{}".format(mq.jq.jid_r_m) )
    if 'd' in sching_m:
      r_numj_l = list(range(sching_m['d'] ) )
      for jid, r in mq.jq.jid_r_m.items():
        r_numj_l[r] += 1
      r_numj_l[0] = T - sum(r_numj_l)
      r_freqj_l = [nj/T for nj in r_numj_l]
      print("r_freqj_l= {}".format(r_freqj_l) )
      d_freqj_l = [nj/T for nj in mq.d_numj_l]
      print("d_freqj_l= {}".format(d_freqj_l) )
      
      sl2 += sl**2
  return sl/nf, sl2/nf

def plot_reptod_ifidle_vs_withdraw():
  ns, d = 12, 4
  T = 30000
  ts_dist = TPareto(1, 10**10, 1.1) # TPareto(1, 10, 1), DUniform(1, 1)
  sl_dist = Dolly() # TPareto(1, 20, 1) # Exp(1) # Exp(1, D=1) # DUniform(1, 1)
  alog("ns= {}, d= {}, T= {}, ts_dist= {}, sl_dist= {}".format(ns, d, T, ts_dist, sl_dist) )
  
  ar_l = []
  Esl_ifidle_l, Vsl_ifidle_l = [], []
  Esl_withdraw_l, Vsl_withdraw_l = [], []
  # sl_god_withdraw_l = []
  nf = 1
  # for ar in np.linspace(0.02, 0.5, 8):
  for ar in np.linspace(0.01, 0.17, 8):
    print("\n ar= {}".format(ar) )
    ar_l.append(ar)
    
    sching_m = {'reptod-ifidle': 0, 'd': d}
    sl, sl2 = sim_sl(nf, ar, ns, T, sching_m, ts_dist, sl_dist)
    print("sching_m= {} \n sl= {}, sl2= {}".format(sching_m, sl, sl2) )
    Esl_ifidle_l.append(sl)
    Vsl_ifidle_l.append(sl2 - sl**2)
    
    sching_m = {'reptod-withdraw': 0, 'd': d, 'L': 0}
    sl, sl2 = sim_sl(nf, ar, ns, T, sching_m, ts_dist, sl_dist)
    print("sching_m= {} \n sl= {}, sl2= {}".format(sching_m, sl, sl2) )
    Esl_withdraw_l.append(sl)
    Vsl_withdraw_l.append(sl2 - sl**2)
    
    # sching_m = {'reptogod-withdraw': 0, 'd': d, 'L': 0}
    # sl = sim_sl(nf, ar, ns, T, sching_m)
    # print("sching_m= {} \n sl= {}".format(sching_m, sl) )
    # sl_god_withdraw_l.append(sl)
  plot.plot(ar_l, Esl_ifidle_l, label='reptod-ifidle', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.plot(ar_l, Esl_withdraw_l, label='reptod-withdraw', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  # plot.plot(ar_l, sl_god_withdraw_l, label='reptod-god-withdraw', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.title(r'$n= {}$, $d= {}$'.format(ns, d) )
  plot.xlabel(r'$\lambda$', fonsize=14)
  plot.ylabel(r'E[Sl]', fonsize=14)
  plot.legend()
  plot.savefig("Esl_reptod_ifidle_vs_withdraw.pdf")
  plot.gcf().clear()
  
  plot.plot(ar_l, Vsl_ifidle_l, label='reptod-ifidle', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.plot(ar_l, Vsl_withdraw_l, label='reptod-withdraw', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.title(r'$n= {}$, $d= {}$'.format(ns, d) )
  plot.xlabel(r'$\lambda$', fonsize=14)
  plot.ylabel(r'Var[Sl]', fonsize=14)
  plot.legend()
  plot.savefig("Vsl_reptod_ifidle_vs_withdraw.pdf")
  plot.gcf().clear()
  log(WARNING, "done.")
  
if __name__ == "__main__":
  plot_reptod_ifidle_vs_withdraw()
