import sys
import numpy as np
from mpi4py import MPI
from rvs import *

def ar_ub_reptod_wcancel(ns, J, S):
  EJ, ES = J.mean(), S.mean()
  return ns/EJ/ES

ns, d = 10, 2
J = TPareto(1, 10**4, 1.1) # Exp(0.05, D=1) # HyperExp([0.8, 0.2], [1, 0.1] )
S = Dolly() # TPareto(1, 100, 1.2) # Bern(1, 20, 0.2)
N, T = 30, ns*5000 # ns*2500
L = 30
wjsize = True
wsysload = False # True

s_len = d+1 if wjsize or wsysload else d
sching_opt = 'wjsize' if wjsize else None
a_len, nn_len = 2, 5 # s_len # 100 # 5 # 40 # 20
ar_ub = ar_ub_reptod_wcancel(ns, J, S)
ar_ub = 0.8*2*ar_ub/3
ar_l = [*np.linspace(0.005, 0.8*ar_ub, 3, endpoint=False), *np.linspace(0.8*ar_ub, ar_ub, 3) ]
ar_l = [ar_l[4] ]

from scher import PolicyGradScher
from learn_howtorep import MultiQ_wRep, sample_traj, sim

act_max = False # True
jg_type = 'deterministic' # 'poisson' # 'selfsimilar'

DONOTLEARN = False # True
EJ, ES = J.mean(), S.mean()
EB = EJ*ES
# sching_m_l = [{'name': 'norep', 'd': d, 's_len': d},
#               {'name': 'reptod', 'd': d, 's_len': d},
#               {'name': 'reptod-ifidle', 'd': d, 's_len': d},
#               {'name': 'reptod-ifidle-wcancel', 'd': d, 's_len': d, 'L': 0},
#               {'name': 'reptod-ifidle-wlatecancel', 'd': d, 's_len': d, 'L': EB},
#               {'name': 'reptod-wcancel', 'd': d, 's_len': d, 'L': 0},
#               {'name': 'reptod-wlatecancel', 'd': d, 's_len': d, 'L': EB} ]
sching_m_l = [{'name': 'norep', 'd': d, 's_len': d},
              {'name': 'reptod', 'd': d, 's_len': d},
              {'name': 'reptod-ifidle', 'd': d, 's_len': d} ]

if not DONOTLEARN:
  sching_m_l.append({'name': 'reptod-wlearning', 'd': d, 's_len': s_len, 'opt': sching_opt} )

def plot_eval_wmpi(rank, T):
  if rank == 0:
    sching__Esl_l_l = [[] for _ in sching_m_l]
    for ar in ar_l:
      scher = learn_howtorep_wmpi(rank, ar) if not DONOTLEARN else None
      sching_Esl_l = eval_wmpi(rank, scher, ar, T)
      for s, Esl in enumerate(sching_Esl_l):
        sching__Esl_l_l[s].append(Esl)
    
    for s, sching_m in enumerate(sching_m_l):
      print("scher= {}, Esl_l= \n{}".format(sching_m['name'], sching__Esl_l_l[s] ) )
      plot.plot(ar_l, sching__Esl_l_l[s], label=sching_m['name'], color=next(dark_color), marker=next(marker), linestyle=':')
    plot.legend()
    plot.xlabel(r'$\lambda$', fontsize=13)
    plot.ylabel(r'Average slowdown', fontsize=13)
    plot.title(r'$n= {}$, $d= {}$, $J \sim {}$, $S \sim {}$'.format(ns, d, J.tolatex(), S.tolatex() ) )
    plot.savefig("plot_eval_wmpi.png")
    plot.gcf().clear()
  else:
    for ar in ar_l:
      scher = learn_howtorep_wmpi(rank, ar) if not DONOTLEARN else None
      eval_wmpi(rank, scher, ar, T)
  log(WARNING, "done; rank= {}".format(rank) )

def eval_wmpi(rank, scher, ar, T):
  alog("starting; rank= {}, ar= {}, T= {}".format(rank, ar, T) )
  sys.stdout.flush()
  
  if rank == 0:
    sching_Esl_l = []
    for i, sching_m in enumerate(sching_m_l):
      for n in range(N):
        p = n % (size-1) + 1
        eval_i = np.array([i], dtype='i')
        comm.Send([eval_i, MPI.INT], dest=p)
      
      Esl_l = []
      # cum_sl_l = []
      for n in range(N):
        p = n % (size-1) + 1
        
        Esl = np.empty(1, dtype=np.float64)
        comm.Recv(Esl, source=p)
        Esl_l.append(Esl)
        # sl_l = np.empty(T, dtype=np.float64)
        # comm.Recv(sl_l, source=p)
        # cum_sl_l += sl_l.tolist()
      print("Eval with sching_m= {}".format(sching_m) )
      Esl = np.mean(Esl_l)
      print("Esl= {}".format(Esl) )
      sching_Esl_l.append(Esl if Esl < 200 else None)
      print("\n\n")
      sys.stdout.flush()
      
      # x_l = numpy.sort(cum_sl_l)[::-1]
      # y_l = numpy.arange(x_l.size)/x_l.size
      # plot.step(x_l, y_l, label=sching_m['name'], color=next(dark_color), marker=next(marker), linestyle=':')
    # plot.xscale('log')
    # plot.yscale('log')
    # plot.legend()
    # plot.xlabel(r'Slowdown', fontsize=13)
    # plot.ylabel(r'Tail distribution', fontsize=13)
    # plot.savefig("sltail_ar{0:.2f}.png".format(ar) )
    # plot.gcf().clear()
    
    for p in range(1, size):
      eval_i = np.array([-1], dtype='i')
      comm.Send([eval_i, MPI.INT], dest=p)
      print("Sent req eval_i= {} to p= {}".format(eval_i, p) )
    return sching_Esl_l
  else:
    while True:
      eval_i = np.empty(1, dtype='i')
      comm.Recv([eval_i, MPI.INT], source=0)
      if eval_i == -1:
        return
      
      Esl = sim(ns, sching_m_l[eval_i], scher, J, S, ar, T, act_max, jg_type)
      Esl = np.array([Esl], dtype=np.float64)
      comm.Send([Esl, MPI.FLOAT], dest=0)
      sys.stdout.flush()

def learn_howtorep_wmpi(rank, ar):
  alog("starting; rank= {}, ar= {}".format(rank, ar) )
  scher = PolicyGradScher(s_len, a_len, nn_len, save_name=save_name('log', 'howtorep', ns, d, ar) )
  alog("starting; rank= {}, scher= {}".format(rank, scher) )
  
  global T
  _T = T
  def Ti(i):
    # T0 = 1000
    # return int(min(T0 * 1.1**i, _T) )
    return _T
  
  if rank == 0:
    for i in range(L):
      T = Ti(i)
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
    scher.save(L)
    for p in range(1, size):
      sim_step = np.array([-1], dtype='i')
      comm.Send([sim_step, MPI.INT], dest=p)
      print("Sent req sim_step= {} to p= {}".format(sim_step, p) )
    sys.stdout.flush()
    return scher
  else:
    sching_m = {'name': 'reptod-wlearning', 'd': d, 's_len': s_len, 'opt': sching_opt}
    while True:
      sim_step = np.empty(1, dtype='i')
      comm.Recv([sim_step, MPI.INT], source=0)
      sim_step = sim_step[0]
      if sim_step == -1:
        break
      
      scher.restore(sim_step)
      T = Ti(sim_step)
      t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(ns, sching_m, scher, J, S, ar, T, jg_type)
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      sys.stdout.flush()
    scher.restore(L)
    return scher

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  # print("rank= {}, size= {}".format(rank, size) )
  # comm.Barrier()
  sys.stdout.flush()
  
  alog("rank= {}, ns= {}, d= {}, J= {}, S= {}, wjsize= {}, N= {}, T= {}".format(rank, ns, d, J, S, wjsize, N, T) )
  # for ar in ar_l:
  #   scher = learn_howtorep_wmpi(rank, ar)
  #   eval_wmpi(rank, scher, ar, 10000*ns)
  
  plot_eval_wmpi(rank, 10000*ns)
