import sys
import numpy as np
from mpi4py import MPI

from learn_howtorep import *

def learn_multiple(rank, ns, d):
  for ar in np.linspace(0.1, 2.5, 5):
    learn_howtorep_wmpi(rank, ns, d, ar)

def eval_(ns, d, ar, T, scher):
  sching_m = {'rep-to-d': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)
  sys.stdout.flush()
  
  sching_m = {'rep-to-d-ifidle': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)
  
  sching_m = {'rep-to-d-wlearning': 0, 'd': d}
  print("Eval with sching_m= {}".format(sching_m) )
  for _ in range(3):
    evaluate(ns, ar, T, sching_m, scher)

def learn_howtorep_wmpi(rank, ns, d, ar):
  s_len, a_len = d, d
  nn_len = 10
  scher = PolicyGradScher(s_len, a_len, nn_len, save_name=save_name('log', 'howtorep', ns, d, ar) )
  N, T = 20, 1000*2
  alog("starting; rank= {}, ar= {}, ns= {}".format(rank, ar, ns) )
  
  if rank == 0:
    # alog("BEFORE learning; ar= {}".format(ar) )
    # eval_(ns, d, ar, 5*T, scher)
    
    for i in range(100):
      scher.save(i)
      n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        p = n % (size-1) + 1
        sim_step = np.array([i], dtype='i')
        comm.Send([sim_step, MPI.INT], dest=p)
      
      for n in range(N):
        p = n % (size-1) + 1
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
      scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      sys.stdout.flush()
    print("AFTER learning; ar= {}".format(ar) )
    eval_(ns, d, ar, 20000, scher)
    
    for p in range(1, size):
      sim_step = np.array([-1], dtype='i')
      comm.Send([sim_step, MPI.INT], dest=p)
      print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
    sys.stdout.flush()
  else:
    sching_m = {'rep-to-d-wlearning': 0, 'd': d}
    while True:
      sim_step = np.empty(1, dtype='i')
      comm.Recv([sim_step, MPI.INT], source=0)
      if sim_step == -1:
        sys.stdout.flush()
        return
      
      scher.restore(sim_step[0] )
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(ns, ar, T, sching_m, scher)
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      sys.stdout.flush()
  
if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  print("rank= {}, size= {}".format(rank, size) )
  # comm.Barrier()
  sys.stdout.flush()
  
  # learn_howtorep_wmpi(rank, ns=6, d=2, ar=0.6)
  # learn_howtorep_wmpi(rank, ns=2, d=2, ar=0.1)
  # learn_howtorep_wmpi(rank, ns=10, d=2, ar=1.5)
  learn_multiple(rank, ns=10, d=2)
