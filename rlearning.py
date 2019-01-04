import math, time, random, scipy
import numpy as np
import tensorflow as tf

from log_utils import *
from sim_objs import *
from sim_objs_lessreal import *

LEARNING_RATE = 0.01 # 0.01 # 0.0001
STATE_LEN = 2 # 3

N, Cap = 20, 10
k = BZipf(1, 10)
R = Uniform(1, 1)
M = 1000
sching_m = {
  'a': 3, 'N': -1,
  'learner': 'QLearner_wTargetNet_wExpReplay',
  'exp_buffer_size': 100*M, 'exp_batch_size': M}
mapping_m = {'type': 'spreading'}

lessreal_sim = True
log(INFO, "lessreal_sim= {}".format(lessreal_sim) )
if lessreal_sim:
  b, beta = 10, 3 # 2
  L = Pareto(b, beta) # TPareto(10, 10**5, 2) # TPareto(10, 10**6, 4)
  a, alpha = 1, 3 # 1, 4
  Sl = Pareto(a, alpha) # Uniform(1, 1)
  
  sinfo_m = {
    'njob': 2000*N, # 10*N,
    'nworker': N, 'wcap': Cap, 'ar': None,
    'k_rv': k, 'reqed_rv': R, 'lifetime_rv': L,
    'straggle_m': {'slowdown': lambda load: Sl.sample() } }
else:
  sinfo_m = {
    'njob': 2000*N,
    'nworker': 5, 'wcap': 10,
    'totaldemand_rv': TPareto(10, 1000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 5, 1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': lambda load: random.uniform(0, 0.01) if random.uniform(0, 1) < 0.4 else 1,
      'straggle_dur_rv': DUniform(10, 100),
      'normal_dur_rv': DUniform(1, 1) } }
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 2/5*ar_ub

## Learned with experience_replay.py
ro__learning_count_m = {
  0.1: 1800,
  0.2: 1725,
  0.3: 1710,
  0.4: 1655,
  0.5: 725,
  0.6: 1590,
  0.7: 715,
  0.8: 1325,
  0.9: 2165}

# from experience_replay import L, k
# D_min, D_max = k.l_l*L.l_l, k.u_l*L.u_l
# blog(D_min=D_min, D_max=D_max)
# def normalize_jdemand(D):
#   # return float(D - D_min)/(D_max - D_min) - 0.5
#   return D/D_max

EL, EL2 = L.mean(), L.moment(2)
StdL = math.sqrt(EL2 - EL**2)
def normalize_lifetime(l):
  return (l - EL)/StdL

Ek, Ek2 = k.mean(), k.moment(2)
Stdk = math.sqrt(Ek2 - Ek**2)
def normalize_k(k):
  return (k - Ek)/Stdk

ED = EL*Ek
ED2 = EL2*Ek2
StdD = math.sqrt(ED2 - ED**2)
def normalize_D(d):
  return (d - ED)/StdD

def state(j, wload_l=None, cluster=None):
  try:
    D = j.totaldemand # j.k
  except AttributeError:
    D = j.k*j.reqed*j.lifetime
    # D = normalize_jdemand(j.k*j.reqed*j.lifetime)
  
  if STATE_LEN == 1:
    return [D]
  elif STATE_LEN == 2:
    # return [D, np.mean(wload_l) ]
    # return [D, j.wait_time]
    return [normalize_D(D), np.mean(wload_l) ]
  elif STATE_LEN == 3:
    # return [D, len(cluster.store.items), np.mean(wload_l) ]
    # return [j.k, normalize_lifetime(j.lifetime), j.wait_time]
    return [normalize_k(j.k), normalize_lifetime(j.lifetime), np.mean(wload_l) ]
  elif STATE_LEN == 4:
    # return [D, len(cluster.store.items), min(wload_l), max(wload_l) ]
    return [D, len(cluster.store.items), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 5:
    return [D, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 6:
    return [D, len(cluster.store.items), min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

def state_(jtotaldemand=None, jk=None, jlifetime=None, jwait_time=None, wload_l=None, cluster_qlen=None):
  # jtotaldemand = normalize_jdemand(jtotaldemand)
  if STATE_LEN == 1:
    return [jtotaldemand]
  elif STATE_LEN == 2:
    # return [jtotaldemand, np.mean(wload_l) ]
    # return [jtotaldemand, jwait_time]
    return [normalize_D(jtotaldemand), np.mean(wload_l) ]
  elif STATE_LEN == 3:
    # return [jtotaldemand, cluster_qlen, np.mean(wload_l) ]
    # return [jk, normalize_lifetime(jlifetime), jwait_time]
    return [normalize_k(jk), normalize_lifetime(jlifetime), np.mean(wload_l) ]
  elif STATE_LEN == 4:
    return [jtotaldemand, cluster_qlen, np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 5:
    return [jtotaldemand, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 6:
    return [jtotaldemand, cluster_qlen, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

def sample_traj(sinfo_m, scher, lessreal_sim=False):
  def reward(slowdown):
    # return 1/slowdown
    # return 10 if slowdown < 1.5 else -10
    
    ## The following allows Q-learning to converge
    # if slowdown < 1.1:
    #   return 10
    # elif slowdown < 1.5:
    #   return 10/slowdown
    # else:
    #   return -slowdown
    
    return -slowdown
    # return -slowdown**2
  
  env = simpy.Environment()
  if lessreal_sim:
    cl = Cluster_LessReal(env, scher=scher, **sinfo_m)
    jg = JobGen_LessReal(env, out=cl, **sinfo_m)
  else:
    cl = Cluster(env, scher=scher, **sinfo_m)
    jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T = sinfo_m['njob']
  try:
    s_len = scher.s_len
  except AttributeError:
    s_len = STATE_LEN
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  
  # t = 0
  # for jid, jinfo_m in sorted(cl.jid_info_m.items(), key=itemgetter(0) ):
  #   # blog(t=t, jid=jid, jinfo_m=jinfo_m)
  #   if 'fate' in jinfo_m and jinfo_m['fate'] == 'finished':
  for t in range(T):
    jinfo_m = cl.jid_info_m[t+1]
    t_s_l[t, :] = jinfo_m['s']
    t_a_l[t, :] = jinfo_m['a']
    sl = (jinfo_m['wait_time'] + jinfo_m['run_time'] )/jinfo_m['expected_run_time']
    t_r_l[t, :] = reward(sl)
    t_sl_l[t, :] = sl
  
  return t_s_l, t_a_l, t_r_l, t_sl_l, \
         np.mean([w.avg_load() for w in cl.w_l] ), \
         0 # sum([1 for _, jinfo_m in cl.jid_info_m.items() if 'fate' in jinfo_m and jinfo_m['fate'] == 'dropped'] )/len(cl.jid_info_m)

def sample_sim(sinfo_m, scher, lessreal_sim=False):
  env = simpy.Environment()
  if lessreal_sim:
    cl = Cluster_LessReal(env, scher=scher, **sinfo_m)
    jg = JobGen_LessReal(env, out=cl, **sinfo_m)
  else:
    cl = Cluster(env, scher=scher, **sinfo_m)
    jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T_l, Sl_l = [], []
  for jid, info in cl.jid_info_m.items():
    if 'fate' in info:
      if info['fate'] == 'finished':
        T = info['wait_time'] + info['run_time']
        T_l.append(T)
        Sl_l.append(T/info['expected_run_time'] )
  
  return {
    'ESl': np.mean(Sl_l),
    'StdSl': np.std(Sl_l),
    'Eload': np.mean([w.avg_load() for w in cl.w_l] ),
    'ET': np.mean(T_l),
    'StdT': np.std(T_l) }

def evaluate(sinfo_m, scher):
  alog("scher= {}".format(scher) )
  for _ in range(3):
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
    print("avg_s= {}, avg_a= {}, avg_r= {}".format(np.mean(t_s_l), np.mean(t_a_l), np.mean(t_r_l) ) )

# #############################################  Learner  ###################################### #
class Learner(object):
  def __init__(self, s_len, a_len, nn_len, save_dir='save', save_suffix=None):
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    self.save_dir = save_dir
    self.save_suffix = save_suffix
    
    self.gamma = 1 # 0.99 # 0.9
    
    self.save_path = None
    self.saver = None
    self.sess = None
  
  def save(self, step):
    if self.save_path is None:
      suffix = '' if self.save_suffix is None else '_' + self.save_suffix
      self.save_path = '{}/{}{}'.format(self.save_dir, self, suffix)
    
    save_path = self.saver.save(self.sess, self.save_path, global_step=step)
    log(WARNING, "saved; ", save_path=save_path)
  
  def restore(self, step, save_suffix=None):
    if save_suffix is not None:
      self.save_path = '{}/{}_{}'.format(self.save_dir, self, save_suffix)
    elif self.save_path is None:
      suffix = '' if self.save_suffix is None else '_' + self.save_suffix
      self.save_path = '{}/{}{}'.format(self.save_dir, self, suffix)
    try:
      save_path = self.saver.restore(self.sess, self.save_path + '-{}'.format(step) )
      log(WARNING, "restored; ", save_path=self.save_path)
      return True
    except:
      log(ERROR, "failed;", save_path=self.save_path)
      return False
