import sys
import numpy as np
from mpi4py import MPI

from learn_howtorep import *

def learn_howtorep_wmpi():
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  print("rank= {}, size= {}".format(rank, size) )
  # comm.Barrier()
  sys.stdout.flush()
  
  ns = 6
  s_len, a_len = ns, ns
  nn_len = 10
  scher = PolicyGradScher(s_len, a_len, nn_len, straj_training=False)
  N, T = 20, 1000
  
  if rank == 0:
    print("BEFORE training")
    sching_m = {'rep-to-idle': 0}
    print("Eval with sching_m= {}".format(sching_m) )
    for _ in range(3):
      evaluate(scher, ns, T, sching_m)
    
    for n in range(1, ns+1):
      sching_m = {'n': n}
      print("Eval with sching_m= {}".format(sching_m) )
      for _ in range(3):
        evaluate(scher, ns, T, sching_m)
    sys.stdout.flush()
    
    for i in range(100*100):
      scher.save(i)
      n_t_s_l, n_t_a_l, n_t_r_l, n_t_sl_l = np.zeros((N, T, s_len)), np.zeros((N, T, 1)), np.zeros((N, T, 1)), np.zeros((N, T, 1))
      for n in range(N):
        p = n % (size-1) + 1
        sim_step = np.array([i], dtype='i')
        comm.Send([sim_step, MPI.INT], dest=p)
        # print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
      
      for n in range(N):
        p = n % (size-1) + 1
        t_s_l = np.empty(T*s_len, dtype=np.float64)
        comm.Recv([t_s_l, MPI.FLOAT], source=p)
        # print("master:: received t_s_l from p= {}, for sim_step= {}".format(p, i) )
        t_a_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_a_l, MPI.FLOAT], source=p)
        # print("master:: received t_a_l from p= {}, for sim_step= {}".format(p, i) )
        t_r_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_r_l, MPI.FLOAT], source=p)
        # print("master:: received t_r_l from p= {}, for sim_step= {}".format(p, i) )
        t_sl_l = np.empty(T, dtype=np.float64)
        comm.Recv([t_sl_l, MPI.FLOAT], source=p)
        # print("master:: received t_sl_l from p= {}, for sim_step= {}".format(p, i) )
        # print("master:: received sim data from p= {}, for sim_step= {}".format(p, i) )
        
        n_t_s_l[n, :] = t_s_l.reshape((T, s_len))
        n_t_a_l[n, :] = t_a_l.reshape((T, 1))
        n_t_r_l[n, :] = t_r_l.reshape((T, 1))
        n_t_sl_l[n, :] = t_sl_l.reshape((T, 1))
      # num_shortest_found = 0
      # for n in range(N):
      #   for t in range(T):
      #     s, a = n_t_s_l[n, t], int(n_t_a_l[n, t][0] )
      #     if s[a] - s.min() < 0.01:
      #       num_shortest_found += 1
      # print("i= {}, avg sl= {}, freq shortest found= {}".format(i, np.mean(n_t_sl_l), num_shortest_found/N/T) )
      print("i= {}, avg sl= {}".format(i, np.mean(n_t_sl_l) ) )
      scher.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
      if i % 10 == 0:
        print("Eval:")
        evaluate(scher, ns, 4*T)
      sys.stdout.flush()
    print("Eval after learning:")
    evaluate(scher, ns, T=40000)
    
    for p in range(1, size):
      sim_step = np.array([-1], dtype='i')
      comm.Send([sim_step, MPI.INT], dest=p)
      print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
    sys.stdout.flush()
  else:
    while True:
      sim_step = np.empty(1, dtype='i')
      comm.Recv([sim_step, MPI.INT], source=0)
      # print("p= {} recved req for sim_step= {}".format(rank, sim_step) )
      if sim_step == -1:
        sys.stdout.flush()
        return
      
      scher.restore(sim_step[0] )
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(scher, ns, T)
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      # print("p= {}, sent sim_step= {} t_s_l".format(rank, sim_step) )
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      # print("p= {} sent sim data for sim_step= {}".format(rank, sim_step) )
      sys.stdout.flush()
  
if __name__ == "__main__":
  learn_howtorep_wmpi()
