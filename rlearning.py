import math, time, random, scipy
import numpy as np
import tensorflow as tf

from log_utils import *
from sim_objs import *
from sim_objs_lessreal import *

LEARNING_RATE = 0.01 # 0.0001
STATE_LEN = 6 # 4
def state(j, wload_l=None, cluster=None):
  try:
    D = j.totaldemand # j.k
  except AttributeError:
    D = j.k*j.reqed*j.lifetime
  if STATE_LEN == 1:
    return [D]
  elif STATE_LEN == 3:
    return [D, min(wload_l), max(wload_l) ]
    # return [D, np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 4:
    # return [D, len(cluster.store.items), min(wload_l), max(wload_l) ]
    return [D, len(cluster.store.items), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 5:
    return [D, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 6:
    return [D, len(cluster.store.items), min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

def state_(jtotaldemand, wload_l=None, cluster_qlen=None):
  if STATE_LEN == 1:
    return [jtotaldemand]
  elif STATE_LEN == 3:
    return [jtotaldemand, min(wload_l), max(wload_l) ]
  elif STATE_LEN == 4:
    return [jtotaldemand, cluster_qlen, np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 5:
    return [jtotaldemand, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]
  elif STATE_LEN == 6:
    return [jtotaldemand, cluster_qlen, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]

def sample_traj(sinfo_m, scher, use_lessreal_sim=False):
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
    
    # return -slowdown
    return -slowdown**2
  
  env = simpy.Environment()
  if use_lessreal_sim:
    cl = Cluster_LessReal(env, scher=scher, **sinfo_m)
    jg = JobGen_LessReal(env, out=cl, **sinfo_m)
  else:
    cl = Cluster(env, scher=scher, **sinfo_m)
    jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T = sinfo_m['njob']
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, scher.s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  
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

def evaluate(sinfo_m, scher):
  alog("scher= {}".format(scher) )
  for _ in range(3):
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
    print("avg_s= {}, avg_a= {}, avg_r= {}".format(np.mean(t_s_l), np.mean(t_a_l), np.mean(t_r_l) ) )

# #############################################  Learner  ###################################### #
class Learner(object):
  def __init__(self, s_len, a_len, nn_len, save_dir='save'):
    self.s_len = s_len
    self.a_len = a_len
    self.nn_len = nn_len
    self.save_dir = save_dir
    
    self.gamma = 0.99 # 0.9
    
    self.saver = None
    self.sess = None
  
  def save(self, step):
    save_name = '{}/{}'.format(self.save_dir, self)
    save_path = self.saver.save(self.sess, save_name, global_step=step)
    log(WARNING, "saved; ", save_path=save_path)
  
  def restore(self, step):
    save_name = '{}/{}-{}'.format(self.save_dir, self, step)
    try:
      save_path = self.saver.restore(self.sess, save_name)
      # log(WARNING, "restored; ", save_path=save_path)
      return True
    except:
      return False
