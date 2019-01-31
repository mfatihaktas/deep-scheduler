from sim_objs_wrelaunch import *
from modeling import *

def sim(sinfo_m, mapping_m, sching_m):
  env = simpy.Environment()
  cl = Cluster_wrelaunch(env, scher=Scher_wrelaunch(mapping_m, sching_m), **sinfo_m)
  jg = JobGen_wrelaunch(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  avg_schedload_l = []
  for i, w in enumerate(cl.w_l):
    print("w._id= {}, w.avg_load= {}".format(w._id, w.avg_load() ) )
    avg_schedload_l.append(w.avg_load() )
  
  njobs, njobs_relaunched = 0, 0
  sl_l, serv_sl_l = [], []
  for jid, info in cl.jid_info_m.items():
    if 'fate' in info:
      fate = info['fate']
      if fate == 'finished':
        serv_sl_l.append(info['run_time']/info['expected_run_time'] )
        sl_l.append(
          (info['wait_time'] + info['run_time'] )/info['expected_run_time'] )
    if 'relaunched' in info:
      njobs_relaunched += 1
    njobs += 1
  log(INFO, "", njobs=njobs, njobs_relaunched=njobs_relaunched)
  
  return {
    'sl_mean': np.mean(sl_l),
    'sl_std': np.std(sl_l),
    'serv_sl_mean': np.mean(serv_sl_l),
    'load_mean': np.mean(avg_schedload_l) }

def simple_relaunch_time(j):
  return 2*j.lifetime

def opt_relaunch_time(j):
  ESl = ET_k_n_pareto(j.k, j.k, Sl.loc, Sl.a)
  if ESl > 4:
    return math.sqrt(j.lifetime*ESl)
  else:
    return None

if __name__ == '__main__':
  N, Cap = 20, 10
  k = BZipf(1, 10)
  R = Uniform(1, 1)
  b, beta = 10, 3
  L = Pareto(b, beta)
  a, alpha = 1, 3
  Sl = Pareto(a, alpha)
  ro = 0.3
  sching_m = {'relaunch_time': simple_relaunch_time} # opt_relaunch_time
  mapping_m = {'type': 'spreading'}
  sinfo_m = {
    'njob': 2, # 10*N, # 2000*N,
    'ar': ar_for_ro(ro, N, Cap, k, R, L, Sl),
    'nworker': N, 'wcap': Cap,
    'k_rv': k, 'reqed_rv': R, 'lifetime_rv': L,
    'straggle_m': {'slowdown': lambda load: Sl.sample() } }
  log(INFO, "", sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  sim_m = sim(sinfo_m, mapping_m, sching_m)
  log(INFO, "", sim_m=sim_m)
