import pprint
import numpy as np

from patch import *
from rvs import *
from sim import *
from scher import PolicyGradScher
# from reptod_wcancel import ar_ub_reptod_wcancel
# from profile_scher import plot_scher # causing an import loop

def ar_ub_reptod_wcancel(ns, J, S):
  EJ, ES = J.mean(), S.mean()
  return ns/EJ/ES

class MultiQ_wRep(object):
  def __init__(self, env, n, max_numj, sching_m, scher, S, act_max=False):
    self.env = env
    self.n = n
    self.max_numj = max_numj
    self.sching_m = sching_m
    self.scher = scher
    self.act_max = act_max
    
    self.jq = JQ(env, list(range(self.n) ), self)
    self.q_l = [FCFS(i, env, S, out=self.jq) for i in range(self.n) ]
    self.jid_info_m = {}
    
    self.store = simpy.Store(env)
    self.action = env.process(self.run() )
    
    self.num_jcompleted = 0
    
    self.sysload = 0
  
  def __repr__(self):
    return "MultiQ_wRep[n= {}]".format(self.n)
  
  def get_sorted_d(self, wjsize):
    # return np.array([q.length() for q in self.q_l] )
    i_l = random.sample(range(self.n), self.sching_m['d'] )
    ql_i_l = sorted([(self.q_l[i].length(wjsize), i) for i in i_l] )
    i_l = [ql_i[1] for ql_i in ql_i_l]
    ql_l = [ql_i[0] for ql_i in ql_i_l]
    
    # ql_l = np.array(ql_l)/sum(ql_l)
    return i_l, ql_l
  
  def run(self):
    while True:
      j = (yield self.store.get() )
      wjsize = True if 'opt' in self.sching_m and self.sching_m['opt'] == 'wjsize' else False
      i_l, ql_l = self.get_sorted_d(wjsize)
      s = ql_l + [j.size] if wjsize else ql_l
      # s = ql_l + [self.sysload] if self.sching_m['s_len'] == len(ql_l) + 1 else ql_l
      sname = self.sching_m['name']
      if sname == 'reptod' or sname == 'reptod-wcancel' or sname == 'reptod-wlatecancel':
        toi_l = i_l
      elif sname == 'norep':
        toi_l = i_l[:1]
      elif sname == 'reptod-ifidle' or sname == 'reptod-ifidle-wcancel' or sname == 'reptod-ifidle-wlatecancel':
        i = 0
        while i < len(ql_l) and ql_l[i] == 0: i += 1
        toi_l = i_l[:i] if i > 0 else i_l[:1]
      elif sname == 'reptod-wlearning':
        a = self.scher.get_random_action(s) if not self.act_max else self.scher.get_max_action(s)
        toi_l = i_l if a > 0 else i_l[:1]
      
      a = (len(toi_l) > 1)
      if sname == 'reptod-wcancel' or sname == 'reptod-ifidle-wcancel':
        self.q_l[toi_l[0]].put(Task(j._id, j.k, j.size) )
        for i in toi_l[1:]:
          self.q_l[i].put(Task(j._id, j.k, j.size, type_='r', L=self.sching_m['L'] ) )
      elif sname == 'reptod-wlatecancel' or sname == 'reptod-ifidle-wlatecancel':
        self.q_l[toi_l[0]].put(Task(j._id, j.k, j.size) )
        for i in toi_l[1:]:
          self.q_l[i].put(Task(j._id, j.k, j.size, type_='s', L=self.sching_m['L'] ) )
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
    self.jid_info_m[jid]['sl'] = (self.env.now - jinfo['ent'] )/jinfo['ts']
    self.jid_info_m[jid]['T'] = self.env.now - jinfo['ent']
    
    for i in jinfo['qid_l']:
      if i not in m['deped_from']:
        self.q_l[i].put_c({'m': 'cancel', 'jid': jid} )
    
    self.num_jcompleted += 1
    if self.num_jcompleted > self.max_numj:
      self.env.exit()

def sample_traj(ns, sching_m, scher, J, S, ar, T, jg_type='poisson'):
  # pi = math.pi
  # s = 10*pi
  # def complex_r(sl):
  #   if sl > 99: return -500
  #   else: return 10*math.tan(-(sl - (1 - pi/2*s) )/s)
  reward = lambda sl: -(sl - 1)**(1.1) # (50 - sl) # 1/sl
  
  env = simpy.Environment()
  jg = JG(env, ar, DUniform(1, 1), J, T, jg_type)
  mq = MultiQ_wRep(env, ns, T, sching_m, scher, S)
  jg.out = mq
  jg.init()
  env.run()
  
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, sching_m['s_len'] )), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  for t in range(T):
    jinfo_m = mq.jid_info_m[t+1]
    # print("t= {}, jinfo_m= {}".format(t, jinfo_m) )
    t_s_l[t, :] = jinfo_m['s']
    t_a_l[t, :] = jinfo_m['a']
    t_r_l[t, :] = reward(jinfo_m['sl'] )
    t_sl_l[t, :] = jinfo_m['sl']
  return t_s_l, t_a_l, t_r_l, t_sl_l

def train(ns, sching_m, scher, J, S, ar, N, T):
  n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
  for n in range(N):
    s_l, a_l, r_l, sl_l = sample_traj(ns, sching_m, scher, J, S, ar, T)
    n_t_s_l[n, :] = s_l
    n_t_a_l[n, :] = a_l
    n_t_r_l[n, :] = r_l
    n_t_sl_l[n, :] = sl_l
  print("avg a= {}, avg sl= {}".format(np.mean(n_t_a_l), np.mean(n_t_sl_l) ) )
  scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)

def evaluate(ns, sching_m, scher, J, S, ar, T):
  _, t_a_l, _, t_sl_l = sample_traj(ns, sching_m, scher, J, S, ar, T)
  Esl = np.mean(t_sl_l)
  print("avg a= {}, avg sl= {}".format(np.mean(t_a_l), Esl) )
  return Esl

def sim(ns, sching_m, scher, J, S, ar, T, act_max=False, jg_type='poisson'):
  env = simpy.Environment()
  jg = JG(env, ar, DUniform(1, 1), J, T, jg_type)
  mq = MultiQ_wRep(env, ns, T, sching_m, scher, S, act_max)
  jg.out = mq
  jg.init()
  env.run()
  
  # sl_l = [mq.jid_info_m[t+1]['sl'] for t in range(T) ]
  Esl = np.mean([mq.jid_info_m[t+1]['sl'] for t in range(T) ] )
  ET = np.mean([mq.jid_info_m[t+1]['T'] for t in range(T) ] )
  print("Esl= {}, ET= {}".format(Esl, ET) )
  return Esl

ns, d = 5, 2
J = HyperExp([0.8, 0.2], [1, 0.1] ) # TPareto(1, 10**10, 1.1)
S = Bern(1, 20, 0.2) # Dolly() # DUniform(1, 1)
N, T = 10, 1000
wjsize = False # True
wsysload = True

s_len = d+1 if wjsize or wsysload else d
a_len, nn_len = 2, 10
ar_ub = 0.9*ar_ub_reptod_wcancel(ns, J, S)
ar_l = np.linspace(0.01, ar_ub, 6)

def learn_howtorep(ar):
  alog("ns= {}, d= {}, N= {}, T= {}, J= {}, S= {}, ar= {}".format(ns, d, N, T, J, S, ar) )
  scher = PolicyGradScher(s_len, a_len, nn_len)
  
  alog("BEFORE training")
  sching_m = {'reptod': 0, 'd': d, 's_len': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, sching_m, scher, J, S, ar, T)
  sching_m = {'reptod-ifidle': 0, 'd': d, 's_len': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, sching_m, scher, J, S, ar, T)
  
  sching_m = {'reptod-wlearning': 0, 'd': d, 's_len': s_len}
  for i in range(100*2):
    print("training; i= {}".format(i) )
    train(ns, sching_m, scher, J, S, ar, N, T)
    # if i % 20 == 0:
    #   plot_scher(i, scher, J)
  
  alog("AFTER training")
  T_ = 40*T
  evaluate(ns, sching_m, scher, J, S, ar, T_)
  
  sching_m = {'reptod-ifidle': 0, 'd': d, 's_len': d}
  print("Eval with sching_m= {}".format(sching_m) )
  evaluate(ns, sching_m, scher, J, S, ar, T_)
  
  for d_ in range(1, ns+1):
    sching_m = {'reptod': 0, 'd': d_, 's_len': d_}
    print("Eval with sching_m= {}".format(sching_m) )
    evaluate(ns, sching_m, scher, J, S, ar, T_)

def plot_reptod_compare():
  alog("ns= {}, d= {}, J= {}, S= {}, ar_ub= {}".format(ns, d, J, S, ar_ub) )
  scher = None
  T_ = 50000*3
  
  Esl_reptod_l, Esl_ifidle_l, Esl_wlearning_l, Esl_wcancel_l = [], [], [], []
  for ar in ar_l:
    print("> ar= {}".format(ar) )
    
    sching_m = {'reptod': 0, 'd': d, 's_len': d}
    print("Eval with sching_m= {}".format(sching_m) )
    Esl_reptod_l.append(evaluate(ns, sching_m, scher, J, S, ar, T_) )
    
    sching_m = {'reptod-ifidle': 0, 'd': d, 's_len': d}
    print("Eval with sching_m= {}".format(sching_m) )
    Esl_ifidle_l.append(evaluate(ns, sching_m, scher, J, S, ar, T_) )
    
    # sching_m = {'reptod-wlearning': 0, 'd': d, 's_len': s_len}
    # scher = PolicyGradScher(s_len, a_len, nn_len, straj_training=False)
    # for i in range(50):
    #   print("training; i= {}".format(i) )
    #   train(ns, sching_m, scher, J, S, ar, N, T)
    # print("Eval with sching_m= {}".format(sching_m) )
    # Esl_wlearning_l.append(evaluate(ns, sching_m, scher, J, S, ar, T_) )
    
    sching_m = {'reptod-wcancel': 0, 'd': d, 's_len': d}
    print("Eval with sching_m= {}".format(sching_m) )
    Esl_wcancel_l.append(evaluate(ns, sching_m, scher, J, S, ar, T_) )
  
  plot.plot(ar_l, Esl_reptod_l, label='Reptod', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.plot(ar_l, Esl_ifidle_l, label='Reptod-ifidle', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  # plot.plot(ar_l, Esl_wlearning_l, label='Reptod-wlearning', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.plot(ar_l, Esl_wcancel_l, label='Reptod-wcancel', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot.title(r'$n= {}$, $d= {}$'.format(ns, d) + "\n" + r'$J \sim {}$, $S \sim {}$'.format(J, S) )
  plot.xlabel(r'$\lambda$', fontsize=14)
  plot.ylabel(r'Average slowdown', fontsize=14)
  plot.legend()
  plot.savefig("plot_reptod_compare.pdf")
  plot.gcf().clear()
  alog("done.")

if __name__ == "__main__":
  # for ar in ar_l:
  #   learn_howtorep(ar)
  
  plot_reptod_compare()
