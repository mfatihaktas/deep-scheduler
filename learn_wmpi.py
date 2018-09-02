import sys
import numpy as np
from mpi4py import MPI

from rvs import *
from scheduler import *

def learn_wmpi(rank):
  scher = RLScher(sinfo_m, sching_m)
  N, T, s_len = scher.N, scher.T, scher.s_len
  log(INFO, "starting;", rank=rank, scher=scher)
  sys.stdout.flush()
  
  if rank == 0:
    blog(sinfo_m=sinfo_m)
    for i in range(L):
      scher.save(i)
      n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        p = n % (num_mpiprocs-1) + 1
        sim_step = np.array([i], dtype='i')
        comm.Send([sim_step, MPI.INT], dest=p)
      
      for n in range(N):
        p = n % (num_mpiprocs-1) + 1
        t_s_l = np.empty(T*s_len, dtype=np.float64)
        comm.Recv([t_s_l, MPI.FLOAT], source=p)
        t_a_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_a_l, MPI.FLOAT], source=p)
        t_r_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_r_l, MPI.FLOAT], source=p)
        t_sl_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_sl_l, MPI.FLOAT], source=p)
        
        n_t_s_l[n, :] = t_s_l.reshape((T, s_len))
        n_t_a_l[n, :] = t_a_l.reshape((T, 1))
        n_t_r_l[n, :] = t_r_l.reshape((T, 1))
        n_t_sl_l[n, :] = t_sl_l.reshape((T, 1))
      alog("i= {}, avg a= {}, avg sl= {}".format(i, np.mean(n_t_a_l), np.mean(n_t_sl_l) ) )
      scher.learner.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      sys.stdout.flush()
    scher.save(L)
    for p in range(1, num_mpiprocs):
      sim_step = np.array([-1], dtype='i')
      comm.Send([sim_step, MPI.INT], dest=p)
      print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
    sys.stdout.flush()
    return scher
  else:
    while True:
      sim_step = np.empty(1, dtype='i')
      comm.Recv([sim_step, MPI.INT], source=0)
      sim_step = sim_step[0]
      if sim_step == -1:
        break
      
      scher.restore(sim_step)
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      sys.stdout.flush()
    scher.restore(L)
    return scher

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
  sinfo_m = {
    'njob': 1000, 'nworker': 10, 'wcap': 10,
    'totaldemand_rv': TPareto(1, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 10, 1.1),
    'k_rv': DUniform(1, 1),
    'func_slowdown': slowdown}
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 3/4*ar_ub
  sching_m = {'N': 10}
  L = 150 # number of learning steps
  
  learn_wmpi(rank)
  