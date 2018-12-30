import sys, time
import numpy as np
from mpi4py import MPI

from scheduler import *
from modeling import *

def eval_wmpi(rank):
  log(INFO, "starting;", rank=rank)
  if rank == 0:
    blog(sinfo_m=sinfo_m)
    ro__scheri__ET_l_l_m, ro__scheri__StdT_l_l_m = {}, {}
    ro__scheri__ESl_l_l_m, ro__scheri__StdSl_l_l_m = {}, {}
    ro__scheri__Eload_l_l_m = {}
  sys.stdout.flush()
  
  for ro in ro_l:
    sinfo_m['ar'] = ar_for_ro(ro, N, Cap, k, R, L, Sl)
    if rank == 0:
      scheri__ET_l_l, scheri__StdT_l_l = [], []
      scheri__ESl_l_l, scheri__StdSl_l_l = [], []
      scheri__Eload_l_l = []
      for i, scher in enumerate(scher_l):
        # log(INFO, "Master will send for scher_i= {}".format(i) )
        # sys.stdout.flush()
        for p in range(1, num_mpiprocs):
          scher_i = np.array([i], dtype='i')
          comm.Send([scher_i, MPI.INT], dest=p)
        
        ET_l, StdT_l = [], []
        ESl_l, StdSl_l = [], []
        Eload_l = []
        for p in range(1, num_mpiprocs):
          ET_StdT_ESl_StdSl_Eload = np.empty(5, dtype=np.float64)
          comm.Recv(ET_StdT_ESl_StdSl_Eload, source=p)
          
          ET_l.append(ET_StdT_ESl_StdSl_Eload[0] )
          StdT_l.append(ET_StdT_ESl_StdSl_Eload[1] )
          ESl_l.append(ET_StdT_ESl_StdSl_Eload[2] )
          StdSl_l.append(ET_StdT_ESl_StdSl_Eload[3] )
          Eload_l.append(ET_StdT_ESl_StdSl_Eload[4] )
        log(INFO, "Master; ro= {}".format(ro), scher=scher, \
          ET_l=ET_l, StdT_l=StdT_l, ESl_l=ESl_l, StdSl_l=StdSl_l, Eload_l=Eload_l)
        sys.stdout.flush()
        scheri__ET_l_l.append(ET_l)
        scheri__StdT_l_l.append(StdT_l)
        scheri__ESl_l_l.append(ESl_l)
        scheri__StdSl_l_l.append(StdSl_l)
        scheri__Eload_l_l.append(Eload_l)
      ro__scheri__ET_l_l_m[ro] = scheri__ET_l_l
      ro__scheri__StdT_l_l_m[ro] = scheri__StdT_l_l
      ro__scheri__ESl_l_l_m[ro] = scheri__ESl_l_l
      ro__scheri__StdSl_l_l_m[ro] = scheri__StdSl_l_l
      ro__scheri__Eload_l_l_m[ro] = scheri__Eload_l_l
      # time.sleep(2)
        
      for p in range(1, num_mpiprocs):
        scher_i = np.array([-1], dtype='i')
        comm.Send([scher_i, MPI.INT], dest=p)
        print("Sent req scher_i= {} to p= {}".format(scher_i, p) )
    else:
      # log(INFO, "rank= {} waiting for Master".format(rank) )
      # sys.stdout.flush()
      while True:
        scher_i = np.empty(1, dtype='i')
        comm.Recv([scher_i, MPI.INT], source=0)
        scher_i = scher_i[0]
        if scher_i == -1:
          break
        
        scher = scher_l[scher_i]
        # if scher_i == 0:
        #   scher.restore(ro__learning_count_m[ro] )
        log(INFO, "rank= {} will sim with scher= {}".format(rank, scher) )
        sys.stdout.flush()
        sim_m = sample_sim(sinfo_m, scher, lessreal_sim)
        log(INFO, "rank= {}".format(rank), sim_m=sim_m, scher=scher)
        
        l = np.array([sim_m['ET'], sim_m['StdT'], sim_m['ESl'], sim_m['StdSl'], sim_m['Eload'] ], dtype=np.float64)
        comm.Send([l, MPI.FLOAT], dest=0)
        sys.stdout.flush()
  if rank == 0:
    blog(scher_l=scher_l, \
      ro__scheri__ET_l_l_m=ro__scheri__ET_l_l_m, ro__scheri__StdT_l_l_m=ro__scheri__StdT_l_l_m, \
      ro__scheri__ESl_l_l_m=ro__scheri__ESl_l_l_m, ro__scheri__StdSl_l_l_m=ro__scheri__StdSl_l_l_m, \
      ro__scheri__Eload_l_l_m=ro__scheri__Eload_l_l_m)

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
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
    b, beta = 10, 4
    L = Pareto(b, beta)
    a, alpha = 1, 3
    Sl = Pareto(a, alpha)
    
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
    # ar_ub = arrival_rate_upperbound(sinfo_m)
    # sinfo_m['ar'] = 2/5*ar_ub
  
  # ro__learning_count_m = {
  #   0.3: 170,
  #   0.5: None,
  #   0.6: 280,
  #   0.75: 520,
  #   0.85: None}
  
  # ro_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  ro_l = [0.2, 0.5, 0.8]
  r = 2
  l, u = L.l_l*Sl.l_l, 40*L.mean()*Sl.mean()
  # d_l = [0, *np.logspace(math.log10(l), math.log10(u), 20) ]
  d_l = [0, *np.logspace(math.log10(l), math.log10(u), 6) ]
  # scher = RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay_persist')
  scher_l = [
    # scher, 
    # Scher(mapping_m, {'type': 'plain', 'a': 0} ),
    # Scher(mapping_m, {'type': 'plain', 'a': sching_m['a'] } ),
    *[Scher_wMultiplicativeExpansion(mapping_m, {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': d} ) for d in d_l]
  ]
  # {'type': 'expand_if_totaldemand_leq', 'threshold': 100, 'a': 1}
  
  eval_wmpi(rank)
