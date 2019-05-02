import numpy as np
import concurrent.futures
from operator import itemgetter

from sim_exp import arrival_rate_upperbound
from mapper import *
from policygrad_learning import *
from q_learning import *

# ############################################  Scher  ########################################### #
class Scher(object):
  def __init__(self, mapping_m, sching_m):
    self.sching_m = sching_m
    self.mapper = Mapper(mapping_m)
    
    self._type = 'Scher'
    self._id = 'Scher_a={}'.format(sching_m['a'] )
    self.s_len = 1
    
    if sching_m['type'] == 'plain':
      self.schedule = self.plain
    elif sching_m['type'] == 'expand_if_totaldemand_leq':
      self.schedule = self.expand_if_totaldemand_leq
    elif sching_m['type'] == 'opportunistic':
      self.schedule = self.opportunistic
  
  def __repr__(self):
    return 'Scher[sching_m={}, mapper= {}]'.format(self.sching_m, self.mapper)
  
  def plain(self, j, w_l, cluster, expand=True):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    
    a = self.sching_m['a'] if expand else 0
    # j.n = min(int(j.k + a), len(w_l) )
    j.n = int(j.k + a)
    if len(w_l) < j.n:
      return None, -1, None
    return None, a, w_l[:j.n]
  
  def expand_if_totaldemand_leq(self, j, w_l, cluster):
    try:
      D = j.totaldemand
    except AttributeError: # use_lessreal_sim = True
      D = j.k*j.reqed*j.lifetime
    
    expand = True if D < self.sching_m['threshold'] else False
    return self.plain(j, w_l, cluster, expand)
  
  def opportunistic(self, j, w_l, cluster):
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
    return None, 1, w_l_[:int(j.k + self.sching_m['a'] ) ]

# #################################  Scher_wMultiplicativeExpansion  ############################# #
class Scher_wMultiplicativeExpansion(object):
  def __init__(self, mapping_m, sching_m):
    self.sching_m = sching_m
    self.mapper = Mapper(mapping_m)
    
    self._type = 'Scher_wMultiplicativeExpansion'
    if sching_m['type'] == 'plain':
      self.schedule = self.plain
    elif sching_m['type'] == 'expand_if_totaldemand_leq':
      self.schedule = self.expand_if_totaldemand_leq
      self._id = 'd={}'.format(self.sching_m['threshold'] )
  
  def __repr__(self):
    return 'Scher_wMultiplicativeExpansion[sching_m={}, mapper= {}, _id= {}]'.format(self.sching_m, self.mapper, self._id)
  
  def plain(self, j, w_l, cluster, expand=True):
    r = self.sching_m['r'] if expand else 1
    j.n = int(j.k*r)
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.n:
      return None, -1, None
    return None, r, w_l[:j.n]
  
  def expand_if_totaldemand_leq(self, j, w_l, cluster):
    expand = True if j.k*j.reqed*j.lifetime < self.sching_m['threshold'] else False
    return self.plain(j, w_l, cluster, expand)

# ###########################################  RLScher  ########################################## #
NN_len = 20 # 10
class RLScher():
  def __init__(self, sinfo_m, mapping_m, sching_m, save_dir='save', save_suffix=None):
    self.sinfo_m = sinfo_m
    
    self._type = 'RLScher'
    self._id = 'RLScher_{}'.format(save_suffix)
    self.mapper = Mapper(mapping_m)
    
    self.s_len = STATE_LEN
    self.a_len = sching_m['a'] + 1
    self.N, self.T = sching_m['N'], sinfo_m['njob']
    
    if sching_m['learner'] == 'PolicyGradLearner':
      self.learner = PolicyGradLearner(self.s_len, self.a_len, nn_len=NN_len, w_actorcritic=True, save_dir=save_dir, save_suffix=save_suffix)
    elif sching_m['learner'] == 'QLearner':
      self.learner = QLearner(self.s_len, self.a_len, nn_len=NN_len, save_dir=save_dir, save_suffix=save_suffix)
    elif sching_m['learner'] == 'QLearner_wTargetNet':
      self.learner = QLearner_wTargetNet(self.s_len, self.a_len, nn_len=NN_len, save_dir=save_dir, save_suffix=save_suffix)
    elif sching_m['learner'] == 'QLearner_wTargetNet_wExpReplay':
      self.learner = QLearner_wTargetNet_wExpReplay(self.s_len, self.a_len, exp_buffer_size=sching_m['exp_buffer_size'], exp_batch_size=sching_m['exp_batch_size'], nn_len=NN_len, save_dir=save_dir, save_suffix=save_suffix)
  
  def __repr__(self):
    return 'RLScher[learner= {}]'.format(self.learner)
  
  def save(self, step):
    return self.learner.save(step)
  
  def restore(self, step, save_suffix=None):
    return self.learner.restore(step, save_suffix)
  
  def summarize(self):
    print("////////////////////////////////////////////////////")
    if 'totaldemand_rv' in self.sinfo_m:
      D = self.sinfo_m['totaldemand_rv']
      l, u = D.l_l, D.u_l
      i = u/10
    elif 'reqed_rv' in self.sinfo_m:
      R = self.sinfo_m['reqed_rv']
      L = self.sinfo_m['lifetime_rv']
      l = R.l_l*L.l_l
      u = 500*l
      i = u/10
    logl, logi, logu = math.log10(l), math.log10(i), math.log10(u)
    D_l = list(np.logspace(logl, logi, 5, endpoint=False) ) + list(np.logspace(logi, logu, 5) )
    if STATE_LEN == 1:
      for D in D_l:
      # for D in np.linspace(1, 300, 10):
        qa_l = self.learner.get_a_q_l(state_(D) )
        print("D= {}, qa_l= {}".format(D, qa_l) )
        blog(a=np.argmax(qa_l) )
    elif STATE_LEN == 2:
      for Eload in [0.1, 0.5, 0.9]:
        for D in D_l:
          qa_l = self.learner.get_a_q_l(state_(jtotaldemand=D, wload_l=[Eload]) )
          print("Eload= {}, D= {}, qa_l= {}".format(Eload, D, qa_l) )
          blog(a=np.argmax(qa_l) )
    elif STATE_LEN == 3:
      # for wait_time in [0, 100, 1000, 100000]:
      for Eload in [0.1, 0.5, 0.9]:
        for k in [1, 3, 7]:
          for lifetime in [20, 1000, 10000]:
            # qa_l = self.learner.get_a_q_l(state_(jk=k, jlifetime=lifetime, wait_time=wait_time) )
            # print("wait_time= {}, k= {}, lifetime= {}; qa_l= {}".format(wait_time, k, lifetime, qa_l) )
            qa_l = self.learner.get_a_q_l(state_(jk=k, jlifetime=lifetime, wload_l=[Eload] ) )
            print("Eload= {}, k= {}, lifetime= {}; qa_l= {}".format(Eload, k, lifetime, qa_l) )
            blog(a=np.argmax(qa_l) )
    # elif 3 <= STATE_LEN <= 6:
    #   for wload_l in [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.9, 0.9, 0.9, 0.9] ]:
    #     print(">>> wload_l= {}".format(wload_l) )
    #     for cluster_qlen in [0, 1, 2, 10]:
    #       print(">> cluster_qlen= {}".format(cluster_qlen) )
    #       for D in D_l:
    #         qa_l = self.learner.get_a_q_l(state_(D, wload_l, cluster_qlen) )
    #         print("D= {}, qa_l= {}".format(D, qa_l) )
    #         blog(a=np.argmax(qa_l) )
    print("----------------------------------------------------")
  
  def schedule(self, j, w_l, cluster):
    w_l = self.mapper.worker_l(j, w_l)
    if len(w_l) < j.k:
      return None, -1, None
    # s = state(j, [w.sched_load() for w in w_l], cluster)
    s = state(j, [w.sched_load() for w in cluster.w_l], cluster)
    # log(INFO, "s= {}".format(s) )
    
    a = self.learner.get_random_action(s)
    j.n = int(j.k + a)
    if len(w_l) < j.n:
      return None, -1, None
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
        # self.learner.save(i)

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
