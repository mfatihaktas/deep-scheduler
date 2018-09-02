import numpy as np

from rvs import *
from sim_objs import *

def arrival_rate_upperbound(sinfo_m):
  return sinfo_m['nworker']*sinfo_m['wcap']/sinfo_m['totaldemand_rv'].mean()/sinfo_m['k_rv'].mean()

def offered_load(sinfo_m):
  return round(sinfo_m['ar']*sinfo_m['totaldemand_rv'].mean()*sinfo_m['k_rv'].mean()/sinfo_m['nworker']/sinfo_m['wcap'], 2)

def sim(sinfo_m, mapping_m):
  env = simpy.Environment()
  cl = Cluster(env, mapper=Mapper(mapping_m), scher=Scher(), **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  # env.run(until=sinfo_m['njob']/sinfo_m['ar'] )
  
  avg_schedload_l = []
  for i, w in enumerate(cl.w_l):
    avg_schedload = np.mean(w.sched_load_l)
    print("w._id= {}, mean(w.sched_load_l)= {}".format(w._id, avg_schedload) )
    avg_schedload_l.append(avg_schedload)
  
  njobs_wfate, ndropped = 0, 0
  slowdown_l = []
  for jid, info in cl.jid_info_m.items():
    if 'fate' in info:
      njobs_wfate += 1
      fate = info['fate']
      if fate == 'dropped':
        ndropped += 1
      elif fate == 'finished':
        slowdown_l.append(info['runtime']/info['expected_lifetime'] )
  blog(ndropped=ndropped, njobs_wfate=njobs_wfate)
  
  return {
    'drop_rate': ndropped/len(cl.jid_info_m),
    'avg_slowdown': np.mean(slowdown_l),
    'avg_utilization': np.mean(avg_schedload_l) }

def slowdown(load):
  threshold = 0.3
  if load < threshold:
    return 1
  else:
    p_max = 0.8 # probability of straggling when load is 1
    p = p_max/(math.e**(1-threshold) - 1) * (math.e**(load-threshold) - 1)
    # return 1-load if random.uniform(0, 1) < p else 1
    return 0.1 if random.uniform(0, 1) < p else 1

def exp():
  sinfo_m = {
    'ar': None, 'njob': 40000, 'nworker': 10, 'wcap': 10,
    'totaldemand_rv': TPareto(1, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 10, 1.1),
    'k_rv': DUniform(1, 1),
    'func_slowdown': slowdown}
  mapping_m = {'type': 'spreading'}
  ar_ub = arrival_rate_upperbound(sinfo_m)
  blog(ar_ub=ar_ub, sinfo_m=sinfo_m, mapping_m=mapping_m)
  
  def wrt_ar():
    for ar in np.linspace(ar_ub/3, ar_ub*3/4, 3):
      print("\nar= {}".format(ar) )
      
      sinfo_m['ar'] = ar
      ol = offered_load(sinfo_m)
      print("offered_load= {}".format(ol) )
      
      m = sim(sinfo_m, mapping_m)
      blog(m=m)
  
  def wrt_exprate():
    ar = ar_ub*3/4
    sinfo_m['ar'] = ar
    print("\nar= {}".format(ar) )
    for max_exprate in np.linspace(1, 3, 3):
      print("max_exprate= {}".format(max_exprate) )
      sinfo_m['max_exprate'] = max_exprate
      
      m = sim(sinfo_m, mapping_m)
      blog(m=m)
  
  # wrt_ar()
  wrt_exprate()
  
  log(INFO, "done.")

if __name__ == '__main__':
  exp()
