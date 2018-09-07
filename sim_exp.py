import numpy as np

from rvs import *
from sim_objs import *

def arrival_rate_upperbound(sinfo_m):
  return sinfo_m['nworker']*sinfo_m['wcap']/sinfo_m['totaldemand_rv'].mean()/sinfo_m['k_rv'].mean()

def offered_load(sinfo_m):
  return round(sinfo_m['ar']*sinfo_m['totaldemand_rv'].mean()*sinfo_m['k_rv'].mean()/sinfo_m['nworker']/sinfo_m['wcap'], 2)

def sim(sinfo_m, mapping_m, sching_m):
  env = simpy.Environment()
  cl = Cluster(env, scher=Scher(mapping_m, sching_m), **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  # env.run(until=sinfo_m['njob']/sinfo_m['ar'] )
  
  avg_schedload_l = []
  for i, w in enumerate(cl.w_l):
    print("w._id= {}, w.avg_load= {}".format(w._id, w.avg_load) )
    avg_schedload_l.append(w.avg_load)
  
  njobs_wfate, ndropped = 0, 0
  slowdown_l = []
  for jid, info in cl.jid_info_m.items():
    if 'fate' in info:
      njobs_wfate += 1
      fate = info['fate']
      if fate == 'dropped':
        ndropped += 1
      elif fate == 'finished':
        slowdown_l.append(
          (info['wait_time'] + info['run_time'] )/info['expected_run_time'] )
  blog(ndropped=ndropped, njobs_wfate=njobs_wfate)
  
  return {
    'drop_rate': ndropped/len(cl.jid_info_m),
    'avg_slowdown': np.mean(slowdown_l),
    'avg_utilization': np.mean(avg_schedload_l) }

def slowdown(load):
  # threshold = 0.3
  # if load < threshold:
  #   return 1
  # else:
  #   p_max = 0.8 # probability of straggling when load is 1
  #   p = p_max/(math.e**(1-threshold) - 1) * (math.e**(load-threshold) - 1)
  #   # return 1-load if random.uniform(0, 1) < p else 1
  #   return 0.1 if random.uniform(0, 1) < p else 1
  
  # return np.random.uniform(0.01, 1 - load)
  return np.random.uniform(0.01, 0.1)

def exp():
  sinfo_m = {
    'ar': None, 'njob': 10000, 'nworker': 10, 'wcap': 10,
    'totaldemand_rv': TPareto(1, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 10, 1.1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': slowdown,
      'straggle_dur_rv': TPareto(1, 100, 1.1),
      'normal_dur_rv': TPareto(1, 100, 1.1) }
  }
  mapping_m = {'type': 'spreading'} # {'type': 'packing'}
  sching_m = {'type': 'opportunistic', 'a': 2} # {'type': 'plain', 'a': 0}
  ar_ub = arrival_rate_upperbound(sinfo_m)
  blog(ar_ub=ar_ub, sinfo_m=sinfo_m, mapping_m=mapping_m)
  
  def wrt_ar():
    # for ar in np.linspace(ar_ub/3, ar_ub*3/4, 3):
    for ar in [ar_ub*2/4]:
    # for ar in [0.1]:
      print("\nar= {}".format(ar) )
      
      sinfo_m['ar'] = ar
      ol = offered_load(sinfo_m)
      print("offered_load= {}".format(ol) )
      
      m = sim(sinfo_m, mapping_m, sching_m)
      blog(m=m)
  
  def wrt_exprate():
    ar = ar_ub*2/4
    sinfo_m['ar'] = ar
    print("\nar= {}".format(ar) )
    for a in np.linspace(0, 2, 3):
      print("a= {}".format(a) )
      sching_m['a'] = a
      
      m = sim(sinfo_m, mapping_m, sching_m)
      blog(m=m)
  
  wrt_ar()
  # wrt_exprate()
  
  log(INFO, "done.")

if __name__ == '__main__':
  exp()
