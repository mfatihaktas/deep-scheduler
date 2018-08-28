import numpy as np

from rvs import *
from sim_objs import *

def arrival_rate_upperbound(sim_m):
  return sim_m['nworker']*sim_m['wcap']/sim_m['totaldemand_rv'].mean()/sim_m['k_rv'].mean()

def offered_load(sim_m):
  return round(sim_m['ar']*sim_m['totaldemand_rv'].mean()*sim_m['k_rv'].mean()/sim_m['nworker']/sim_m['wcap'], 2)

def sim(sim_m, sching_m):
  env = simpy.Environment()
  cl = Cluster(env, scher=Scher(sching_m), **sim_m)
  jg = JobGen(env, out=cl, **sim_m)
  env.run(until=cl.wait_for_alljobs)
  # env.run(until=sim_m['njob']/sim_m['ar'] )
  
  avg_schedload_l = []
  for i, w in enumerate(cl.w_l):
    avg_schedload = np.mean(w.sched_load_l)
    print("w._id= {}, mean(w.sched_load_l)= {}".format(w._id, avg_schedload) )
    avg_schedload_l.append(avg_schedload)
  
  ndropped = 0
  slowdown_l = []
  for jid, info in jid_info_m.items():
    fate = info['fate']
    if fate == 'dropped':
      ndropped += 1
    elif fate == 'finished':
      slowdown_l.append(info['runtime']/info['expected_lifetime'] )
  
  return {
    'drop_rate': ndropped/len(jid_info_m),
    'avg_slowdown': np.mean(slowdown_l),
    'avg_utilization': np.mean(avg_schedload_l) }

def plot_wrt_ar():
  sim_m = {
    'ar': None, 'njob': 1000, 'nworker': 10, 'wcap': 10,
    'totaldemand_rv': TPareto(1, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 10, 1.1),
    'k_rv': DUniform(1, 1) }
  sching_m = {'type': 'spreading'}
  blog(sim_m=sim_m, sching_m=sching_m)
  
  ar_ub = arrival_rate_upperbound(sim_m)
  print("ar_ub= {}".format(ar_ub) )
  
  for ar in np.linspace(ar_ub/3, ar_ub*3/4, 3):
    print("\nar= {}".format(ar) )
    
    sim_m['ar'] = ar
    ol = offered_load(sim_m)
    print("offered_load= {}".format(ol) )
    
    m = sim(sim_m, sching_m)
    blog(m=m)

if __name__ == '__main__':
  plot_wrt_ar()
