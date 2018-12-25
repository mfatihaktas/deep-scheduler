import sys
import numpy as np
from mpi4py import MPI

from rvs import *
from scheduler import *

def eval_wmpi(rank):
  log(INFO, "starting;", rank=rank)
  if rank == 0:
    blog(sinfo_m=sinfo_m)
    ro__scheri__Esl_l_l_m = {}
  sys.stdout.flush()
  
  for ro in ro_l:
    sinfo_m['ar'] = ar_for_ro(ro, N, Cap, k, R, L, Sl)
    if rank == 0:
      scheri__Esl_l_l = []
      for i, scher in enumerate(scher_l):
        for p in range(1, num_mpiprocs):
          scher_i = np.array([i], dtype='i')
          comm.Send([scher_i, MPI.INT], dest=p)
        
        Esl_l = []
        for p in range(1, num_mpiprocs):
          Esl = np.empty(1, dtype=np.float64)
          comm.Recv(Esl, source=p)
          Esl_l.append(Esl)
        log(INFO, "\nEval;", scher=scher, Esl_l=Esl_l)
        sys.stdout.flush()
        scheri__Esl_l_l.append(Esl_l)
      ro__scheri__Esl_l_l_m[ro] = scheri__Esl_l_l
        
      # for p in range(1, num_mpiprocs):
      #   scher_i = np.array([-1], dtype='i')
      #   comm.Send([scher_i, MPI.INT], dest=p)
      #   print("Sent req scher_i= {} to p= {}".format(scher_i, p) )
      blog(scher_l=scher_l, ro__scheri__Esl_l_l_m=ro__scheri__Esl_l_l_m)
    else:
      # while True:
      scher_i = np.empty(1, dtype='i')
      comm.Recv([scher_i, MPI.INT], source=0)
      scher_i = scher_i[0]
      # if scher_i == -1:
      #   return
      
      scher = scher_l[scher_i]
      if scher_i == 0:
        scher.restore(ro__learning_count_m[ro] )
      t_s_l, t_a_l, t_r_l, t_sl_l, load_mean, droprate_mean = sample_traj(sinfo_m, scher, use_lessreal_sim)
      
      Esl = np.array([np.mean(t_sl_l) ], dtype=np.float64)
      comm.Send([Esl, MPI.FLOAT], dest=0)
      sys.stdout.flush()

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
  N, Cap = 20, 10
  k = BZipf(1, 5)
  R = Uniform(1, 1)
  M = 1000
  sching_m = {
    'a': 3, 'N': -1,
    'learner': 'QLearner_wTargetNet_wExpReplay',
    'exp_buffer_size': 100*M, 'exp_batch_size': M}
  mapping_m = {'type': 'spreading'}
  
  use_lessreal_sim = True
  log(INFO, "use_lessreal_sim= {}".format(use_lessreal_sim) )
  if use_lessreal_sim:
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
  
  ro__learning_count_m = {
    0.3: 170,
    0.5: None,
    0.6: 280,
    0.75: 520,
    0.85: None}
  
  ro_l = [0.3, 0.5, 0.6, 0.75, 0.85]
  scher = RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay_persist')
  scher_l = [
    scher, 
    Scher(mapping_m, {'type': 'plain', 'a': 0} ),
    Scher(mapping_m, {'type': 'plain', 'a': sching_m['a'] } ) ]
  # {'type': 'expand_if_totaldemand_leq', 'threshold': 100, 'a': 1}
  
  eval_wmpi(rank)
