import numpy as np
import concurrent.futures
from operator import itemgetter

from learning_utils import *
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
  
  def plain(self, j, w_l, expand=False):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    
    a = self.sching_m['a'] if expand else 0
    j.n = min(int(j.k*(a + 1) ), len(w_l) )
    return None, a, w_l[:j.n]
  
  def expand_if_totaldemand_leq(self, j, w_l):
    expand = True if j.totaldemand < self.sching_m['threshold'] else False
    return self.plain(j, w_l, expand)
  
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
class RLScher():
  def __init__(self, sinfo_m, mapping_m, sching_m):
    self.sinfo_m = sinfo_m
    
    self.mapper = Mapper(mapping_m)
    
    self.s_len = STATE_LEN
    self.a_len = sching_m['a'] + 1
    self.N, self.T = sching_m['N'], sinfo_m['njob']
    
    # self.learner = PolicyGradLearner(self.s_len, self.a_len, nn_len=10, w_actorcritic=True)
    self.learner = QLearner(self.s_len, self.a_len, nn_len=10)
  
  def __repr__(self):
    return 'RLScher[learner={}]'.format(self.learner)
  
  def save(self, step):
    return self.learner.save(step)
  
  def restore(self, step):
    return self.learner.restore(step)
  
  def summarize(self):
    job_totaldemand_rv = self.sinfo_m['totaldemand_rv']
    if STATE_LEN == 1:
      for totaldemand in np.logspace(0.1, math.log10(job_totaldemand_rv.u_l/10), 10):
      # for totaldemand in np.linspace(1, 300, 10):
        j = Job(_id=0, k=1, n=1, demandperslot_rv=TNormal(1, 1), totaldemand=totaldemand)
        qa_l = self.learner.get_qa_l(state(j) )
        print("totaldemand= {}, qa_l= {}".format(totaldemand, qa_l) )
    elif STATE_LEN == 3 or STATE_LEN == 5:
      for load1 in np.linspace(0, 1, 5):
        for load2 in np.linspace(load1, 1, 2):
          for totaldemand in np.logspace(0.1, math.log10(job_totaldemand_rv.u_l/10), 10):
            j = Job(_id=0, k=1, n=1, demandperslot_rv=TNormal(1, 1), totaldemand=totaldemand)
            qa_l = self.learner.get_qa_l(state(j, [load1, load2] ) )
            print("load1= {}, load2= {}, totaldemand= {}, qa_l= {}".format(load1, load2, totaldemand, qa_l) )
    
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

if __name__ == '__main__':
  sinfo_m = {
    'njob': 2000, 'nworker': 10, 'wcap': 10, # 10000
    'totaldemand_rv': TPareto(100, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 5, 1.1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': lambda load: np.random.uniform(0.01, 0.1),
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
