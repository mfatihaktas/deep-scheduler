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
  # for jid, info in cl.jid_info_m.items():
  #   if 'fate' in info:
  #     fate = info['fate']
  #     if fate == 'finished':
  #       serv_sl_l.append(info['run_time']/info['expected_run_time'] )
  #       sl_l.append(
  #         (info['wait_time'] + info['run_time'] )/info['expected_run_time'] )
  #   if 'relaunched' in info:
  #     njobs_relaunched += 1
  for jid in range(1, sinfo_m['njob']+1):
    info = cl.jid_info_m[jid]
    serv_sl_l.append(info['run_time']/info['expected_run_time'] )
    sl_l.append(
      (info['wait_time'] + info['run_time'] )/info['expected_run_time'] )
    if 'nrelaunched' in info:
      # print("k= {}, nrelaunched= {}".format(len(info['wid_l'] ), info['nrelaunched'] ) )
      njobs_relaunched += 1
    njobs += 1
  log(INFO, "", njobs=njobs, njobs_relaunched=njobs_relaunched)
  
  return {
    'sl_mean': np.mean(sl_l),
    'sl_std': np.std(sl_l),
    'serv_sl_mean': np.mean(serv_sl_l),
    'load_mean': np.mean(avg_schedload_l) }

def subopt_relaunch_time(j):
  return 4*j.lifetime

def opt_relaunch_time(j):
  ESl = ET_k_n_pareto(j.k, j.k, Sl.loc, Sl.a)
  if ESl > 4:
    return math.sqrt(ESl)*j.lifetime
  else:
    return None
  # return math.sqrt(ESl)*j.lifetime

def print_optimal_d():
  def alpha_gen(ro):
    return alpha
  r = 2
  # print("ESl2= {}".format(Sl.moment(2) ) )
  # print("ESl2_pareto= {}".format(ESl2_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=0, red='Coding') ) )
  
  for ro0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    d_opt = optimal_d_pareto(ro0, N, Cap, k, r, b, beta, a, alpha_gen, red='Coding')
    print("ro0= {}, d_opt= {}".format(ro0, d_opt) )
    
    # ET = ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d=0, red='Coding')[0]
    # print("ro0= {}, ET= {}".format(ro0, ET) )

def exp():
  print(">> OPT:")
  sching_m = {'relaunch_time': opt_relaunch_time}
  sim_m = sim(sinfo_m, mapping_m, sching_m)
  log(INFO, "", sching_m=sching_m, sim_m=sim_m)
  
  print(">> SUB-OPT:")
  sching_m = {'relaunch_time': subopt_relaunch_time}
  sim_m = sim(sinfo_m, mapping_m, sching_m)
  log(INFO, "", sching_m=sching_m, sim_m=sim_m)
  
  print(">> NONE:")
  sching_m = {'relaunch_time': lambda j: None}
  sim_m = sim(sinfo_m, mapping_m, sching_m)
  log(INFO, "", sching_m=sching_m, sim_m=sim_m)

if __name__ == '__main__':
  N, Cap = 20, 10
  k = BZipf(1, 10)
  R = Uniform(1, 1)
  b, beta = 10, 3
  L = Pareto(b, beta)
  a, alpha = 1, 2.05 # 3
  Sl = Pareto(a, alpha)
  ro = 0.3
  log(INFO, "ro= {}".format(ro) )
  mapping_m = {'type': 'spreading'}
  sinfo_m = {
    'njob': 2000*N, # 10*N
    'ar': ar_for_ro(ro, N, Cap, k, R, L, Sl),
    'nworker': N, 'wcap': Cap,
    'k_rv': k, 'reqed_rv': R, 'lifetime_rv': L,
    'straggle_m': {'slowdown': lambda load: Sl.sample() } }
  log(INFO, "", sinfo_m=sinfo_m, mapping_m=mapping_m)
  
  exp()
  
