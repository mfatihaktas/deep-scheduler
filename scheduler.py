import numpy as np
import concurrent.futures
from operator import itemgetter

from sim_objs import *
from sim_exp import arrival_rate_upperbound
from mapper import *
from rlearning import *

# ############################################  Scher  ########################################### #
class Scher(object):
  def __init__(self, mapping_m, sching_m):
    self.sching_m = sching_m
    self.mapper = Mapper(mapping_m)
    
    self.s_len = 1
    
    if sching_m['type'] == 'plain':
      self.schedule = self.plain
    elif sching_m['type'] == 'expand_if_totaldemand_leq':
      self.schedule = self.expand_if_totaldemand_leq
    elif sching_m['type'] == 'opportunistic':
      self.schedule = self.opportunistic
  
  def __repr__(self):
    return 'Scher[sching_m={}, \nmapper= {}]'.format(self.sching_m, self.mapper)
  
  def plain(self, j, w_l):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    a = self.sching_m['a']
    j.n = int(j.k*(a + 1) )
    return None, a, w_l[:j.n]
  
  def expand_if_totaldemand_leq(self, j, w_l):
    if j.totaldemand < self.sching_m['threshold']:
      return self.plain(j, w_l)
    else:
      return None, -1, None
  
  def opportunistic(self, j, w_l):
    if self.sching_m['mapping_type'] == 'packing':
      w_l_ = []
      for w in w_l:
        sched_reqed = sum([t.reqed for t in w.t_l if t.type_ == 's'] )
        if j.reqed <= w.cap - sched_reqed:
          w_l_.append(w)
    elif self.sching_m['mapping_type'] == 'spreading':
      w_load_l = []
      for w in w_l:
        sched_reqed = sum([t.reqed for t in w.t_l if t.type_ == 's'] )
        if j.reqed <= w.cap - sched_reqed:
          w_load_l.append((w, sched_reqed/w.cap) )
      w_l_ = [w for w, _ in sorted(w_load_l, key=itemgetter(1) ) ]
    
    if len(w_l_) < j.k:
      return None, -1, None
    return None, 1, w_l_[:j.k*(self.sching_m['a'] + 1) ]

# ###########################################  RLScher  ########################################## #
STATE_LEN = 5
def state(j, wload_l):
  if STATE_LEN == 1:
    return [j.totaldemand] # j.k
  elif STATE_LEN == 5:
    return [j.totaldemand, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

class RLScher():
  def __init__(self, sinfo_m, mapping_m, sching_m):
    self.sinfo_m = sinfo_m
    
    self.mapper = Mapper(mapping_m)
    
    self.s_len = STATE_LEN
    self.a_len = sching_m['a'] + 1
    self.N, self.T = sching_m['N'], sinfo_m['njob']
    
    # self.learner = PolicyGradLearner(self.s_len, self.a_len, nn_len=10, w_actorcritic=False)
    self.learner = QLearner(self.s_len, self.a_len, nn_len=10)
  
  def __repr__(self):
    return 'RLScher[learner={}]'.format(self.learner)
  
  def save(self, step):
    return self.learner.save(step)
  
  def restore(self, step):
    return self.learner.restore(step)
  
  def schedule(self, j, w_l):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    s = state(j, [w.sched_load() for w in w_l] )
    a = self.learner.get_random_action(s)
    j.n = int(j.k*(a + 1) )
    return s, a, w_l[:j.n]
  
  def train(self, nsteps):
    for i in range(nsteps):
      alog(">> i= {}".format(i) )
      n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((self.N, self.T, self.s_len)), np.zeros((self.N, self.T, 1)), np.zeros((self.N, self.T, 1))
      for n in range(self.N):
        t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(self.sinfo_m, self)
        alog("n= {}, avg_a= {}, avg_r= {}, avg_sl= {}".format(n, np.mean(t_a_l), np.mean(t_r_l), np.mean(t_sl_l) ) )
        n_t_s_l[n], n_t_a_l[n], n_t_r_l[n] = t_s_l, t_a_l, t_r_l
      self.learner.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
  
  def train_multithreaded(self, nsteps):
    if self.learner.restore(nsteps):
      log(WARNING, "learner.restore is a success, will not retrain.")
    else:
      tp = concurrent.futures.ThreadPoolExecutor(max_workers=100)
      for i in range(1, nsteps+1):
        alog(">> i= {}".format(i) )
        n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((self.N, self.T, self.s_len)), np.zeros((self.N, self.T, 1)), np.zeros((self.N, self.T, 1))
        future_n_m = {tp.submit(sample_traj, self.sinfo_m, self): n for n in range(self.N) }
        for future in concurrent.futures.as_completed(future_n_m):
          n = future_n_m[future]
          try:
            t_s_l, t_a_l, t_r_l, t_sl_l = future.result()
          except Exception as exc:
            log(ERROR, "exception;", exc=exc)
          alog("n= {}, avg_a= {}, avg_r= {}, avg_sl= {}".format(n, np.mean(t_a_l), np.mean(t_r_l), np.mean(t_sl_l) ) )
          n_t_s_l[n], n_t_a_l[n], n_t_r_l[n] = t_s_l, t_a_l, t_r_l
        self.learner.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
        self.learner.save(i)

# ############################################  utils  ########################################### #
def sample_traj(sinfo_m, scher):
  def reward(slowdown):
    # return 1/slowdown
    # return 10 if slowdown < 1.5 else -10
    
    if slowdown < 1.1:
      return 10
    elif slowdown < 1.5:
      return 1
    else:
      return -10
  
  env = simpy.Environment()
  cl = Cluster(env, scher=scher, **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T = sinfo_m['njob']
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, scher.s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  
  
  t = 0
  for jid, jinfo_m in sorted(cl.jid_info_m.items(), key=itemgetter(0) ):
    # blog(t=t, jid=jid, jinfo_m=jinfo_m)
    if 'fate' in jinfo_m and jinfo_m['fate'] == 'finished':
      t_s_l[t, :] = jinfo_m['s']
      t_a_l[t, :] = jinfo_m['a']
      sl = (jinfo_m['wait_time'] + jinfo_m['run_time'] )/jinfo_m['expected_run_time']
      t_r_l[t, :] = reward(sl)
      t_sl_l[t, :] = sl
      t += 1
  return t_s_l, t_a_l, t_r_l, t_sl_l, \
         np.mean([w.avg_load for w in cl.w_l] ), \
         sum([1 for _, jinfo_m in cl.jid_info_m.items() if 'fate' in jinfo_m and jinfo_m['fate'] == 'dropped'] )/len(cl.jid_info_m)

def evaluate(sinfo_m, scher):
  alog("scher= {}".format(scher) )
  for _ in range(3):
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
    print("avg_s= {}, avg_a= {}, avg_r= {}".format(np.mean(t_s_l), np.mean(t_a_l), np.mean(t_r_l) ) )

def slowdown(load):
  return np.random.uniform(0.01, 0.1)

if __name__ == '__main__':
  sinfo_m = {
    'njob': 2000, 'nworker': 10, 'wcap': 10, # 10000
    'totaldemand_rv': TPareto(100, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 5, 1.1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': slowdown,
      'straggle_dur_rv': TPareto(10, 100, 1),
      'normal_dur_rv': TPareto(10, 100, 1) } }
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 2/4*ar_ub
  sching_m = {'a': 1, 'N': 10}
  blog(sinfo_m=sinfo_m, sching_m=sching_m)
  
  scher = RLScher(sinfo_m, sching_m)
  # sinfo_m['max_exprate'] = max_exprate
  
  print("scher= {}".format(scher) )
  scher.train_multithreaded(40) # train(40)
  evaluate(sinfo_m, scher)
