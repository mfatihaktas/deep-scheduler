import sys
import numpy as np
from mpi4py import MPI

from learn_howtorep import *
from reptod_wcancel import ar_ub_reptod_wcancel

ns, d = 10, 4
J = TPareto(1, 10**4, 1.1) # Exp(0.1) # HyperExp([0.8, 0.2], [1, 0.1] ) # Exp(0.05, D=1)
S = TPareto(1, 100, 1.2) # Dolly() # Bern(1, 20, 0.2)
N, T = 20, ns*5000 # ns*2500
wjsize = False # True
wsysload = False # True

s_len = d+1 if wjsize or wsysload else d
a_len, nn_len = 2, 10
ar_ub = ar_ub_reptod_wcancel(ns, J, S)
ar_l = [ar for ar in np.linspace(ar_ub/10, 2*ar_ub/3, 5) ] # [ar for ar in np.linspace(1.75*ar_ub/3, 2*ar_ub/3, 3) ] # [ar for ar in np.linspace(ar_ub/3, 2*ar_ub/3, 3) ] # [ar for ar in np.linspace(ar_ub/4, 2*ar_ub/3, 3) ] # [ar for ar in np.linspace(0.01, ar_ub/2, 5) ]

L = 100

act_max = True
jg_type = 'selfsimilar' # 'poisson'

def eval_wmpi(rank, scher, ar, T):
  alog("starting; rank= {}, ar= {}, T= {}".format(rank, ar, T) )
  sys.stdout.flush()
  
  # {'reptod-wlearning': 0, 'd': d, 's_len': s_len},
  sching_m_l = [{'norep': 0, 'd': d, 's_len': d, 'name': 'norep'},
                {'reptod': 0, 'd': d, 's_len': d, 'name': 'reptod'},
                {'reptod-ifidle': 0, 'd': d, 's_len': d, 'name': 'reptod-ifidle'},
                {'reptod-wcancel': 0, 'd': d, 's_len': d, 'name': 'reptod-wcancel'} ]
  if rank == 0:
    for i, sching_m in enumerate(sching_m_l):
      for n in range(N):
        p = n % (size-1) + 1
        eval_i = np.array([i], dtype='i')
        comm.Send([eval_i, MPI.INT], dest=p)
      
      cum_sl_l = []
      for n in range(N):
        p = n % (size-1) + 1
        
        sl_l = np.empty(T, dtype=np.float64)
        comm.Recv(sl_l, source=p)
        cum_sl_l += sl_l.tolist()
      print("Eval with sching_m= {}".format(sching_m) )
      print("Esl= {}".format(np.mean(cum_sl_l) ) )
      print("\n\n")
      sys.stdout.flush()
      
      x_l = numpy.sort(cum_sl_l)[::-1]
      y_l = numpy.arange(x_l.size)/x_l.size
      plot.xscale('log')
      plot.yscale('log')
      plot.step(x_l, y_l, label=sching_m['name'], color=next(dark_color), marker=next(marker), linestyle=':')
    plot.legend()
    plot.xlabel(r'Slowdown', fontsize=13)
    plot.ylabel(r'Tail distribution', fontsize=13)
    plot.savefig("sltail_ar{0:.2f}.png".format(ar) )
    plot.gcf().clear()
    
    
    for p in range(1, size):
      eval_i = np.array([-1], dtype='i')
      comm.Send([eval_i, MPI.INT], dest=p)
      print("Sent req eval_i= {} to p= {}".format(eval_i, p) )
  else:
    while True:
      eval_i = np.empty(1, dtype='i')
      comm.Recv([eval_i, MPI.INT], source=0)
      if eval_i == -1:
        return
      
      sl_l = sim(ns, sching_m_l[eval_i], scher, J, S, ar, T, act_max, jg_type)
      comm.Send([np.array(sl_l), MPI.FLOAT], dest=0)
      sys.stdout.flush()

def learn_howtorep_wmpi(rank, ar):
  alog("starting; rank= {}, ar= {}".format(rank, ar) )
  scher = PolicyGradScher(s_len, a_len, nn_len, save_name=save_name('log', 'howtorep', ns, d, ar) )
  
  if rank == 0:
    for i in range(L):
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
    # eval_(scher, ar, 50000)
    
    for p in range(1, size):
      sim_step = np.array([-1], dtype='i')
      comm.Send([sim_step, MPI.INT], dest=p)
      print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
    sys.stdout.flush()
  else:
    sching_m = {'reptod-wlearning': 0, 'd': d, 's_len': s_len}
    while True:
      sim_step = np.empty(1, dtype='i')
      comm.Recv([sim_step, MPI.INT], source=0)
      if sim_step == -1:
        break
      
      scher.restore(sim_step[0] )
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(ns, sching_m, scher, J, S, ar, T)
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      sys.stdout.flush()
  return scher

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  # print("rank= {}, size= {}".format(rank, size) )
  # comm.Barrier()
  sys.stdout.flush()
  
  alog("rank= {}, ns= {}, d= {}, J= {}, S= {}, wjsize= {}, N= {}, T= {}".format(rank, ns, d, J, S, wjsize, N, T) )
  for ar in ar_l:
    scher = None # learn_howtorep_wmpi(rank, ar)
    eval_wmpi(rank, scher, ar, 10000*ns)
