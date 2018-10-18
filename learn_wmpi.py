import sys
import numpy as np
from mpi4py import MPI

from rvs import *
from scheduler import *

def eval_wmpi(rank):
  log(INFO, "starting;", rank=rank)
  sys.stdout.flush()
  
  if rank == 0:
    blog(sinfo_m=sinfo_m)
    sys.stdout.flush()
    
    schingi_Esl_l = []
    for i, sching_m in enumerate(sching_m_l):
      for p in range(1, num_mpiprocs):
        eval_i = np.array([i], dtype='i')
        comm.Send([eval_i, MPI.INT], dest=p)
      
      Esl_l = []
      # cum_sl_l = []
      for p in range(1, num_mpiprocs):
        Esl = np.empty(1, dtype=np.float64)
        comm.Recv(Esl, source=p)
        Esl_l.append(Esl)
        # sl_l = np.empty(T, dtype=np.float64)
        # comm.Recv(sl_l, source=p)
        # cum_sl_l += sl_l.tolist()
      log(INFO, "\nEval;", sching_m=sching_m, Esl=np.mean(Esl_l) )
      sys.stdout.flush()
      schingi_Esl_l.append(Esl if Esl < 200 else None)
      
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
    
    for p in range(1, num_mpiprocs):
      eval_i = np.array([-1], dtype='i')
      comm.Send([eval_i, MPI.INT], dest=p)
      print("Sent req eval_i= {} to p= {}".format(eval_i, p) )
    return schingi_Esl_l
  else:
    while True:
      eval_i = np.empty(1, dtype='i')
      comm.Recv([eval_i, MPI.INT], source=0)
      eval_i = eval_i[0]
      if eval_i == -1:
        return
      
      scher = Scher(mapping_m, sching_m_l[eval_i] )
      _, _, _, t_sl_l, load_mean, droprate_mean = sample_traj(sinfo_m, scher)
      print("rank= {}, eval_i= {}, slowdown_mean= {}, load_mean= {}, droprate_mean= {}".format(rank, eval_i, np.mean(t_sl_l), load_mean, droprate_mean) )
      
      Esl = np.array([np.mean(t_sl_l) ], dtype=np.float64)
      comm.Send([Esl, MPI.FLOAT], dest=0)
      sys.stdout.flush()

def learn_wmpi(rank):
  scher = RLScher(sinfo_m, mapping_m, sching_m)
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
      alog("i= {}, a_mean= {}, sl_mean= {}, sl_std= {}".format(i, np.mean(n_t_a_l), np.mean(n_t_sl_l), np.std(n_t_sl_l) ) )
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
      t_s_l, t_a_l, t_r_l, t_sl_l, load_mean, droprate_mean = sample_traj(sinfo_m, scher)
      print("rank= {}, sim_step= {}, a_mean= {}, sl_mean= {}, load_mean= {}, droprate_mean= {}".format(rank, sim_step, np.mean(t_a_l), np.mean(t_sl_l), load_mean, droprate_mean) )
      comm.Send([t_s_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_a_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_r_l.flatten(), MPI.FLOAT], dest=0)
      comm.Send([t_sl_l.flatten(), MPI.FLOAT], dest=0)
      sys.stdout.flush()
    scher.restore(L)
    return scher

def slowdown(load):
  # return np.random.uniform(0.01, 0.1)
  threshold = 0.1
  if load < threshold:
    return 1
    # return 0.9 if random.uniform(0, 1) < 0.5 else 1
  else:
    p_max = 0.8 # probability of straggling when load is 1
    p = p_max/(math.e**(1-threshold) - 1) * (math.e**(load-threshold) - 1)
    return 0.1*(1-load) if random.uniform(0, 1) < p else 1

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
  sinfo_m = {
    'njob': 2000, 'nworker': 5, 'wcap': 10,
    'totaldemand_rv': TPareto(10, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 5, 1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': slowdown,
      'straggle_dur_rv': TPareto(10, 1000, 1),
      'normal_dur_rv': TPareto(10, 1000, 1) } }
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 1/2*ar_ub
  mapping_m = {'type': 'spreading'}
  sching_m = {'a': 1, 'N': num_mpiprocs-1}
  L = 100 # number of learning steps
  
  # {'type': 'plain', 'a': 1},
  sching_m_l = [
    {'type': 'plain', 'a': 0},
    {'type': 'expand_if_totaldemand_leq', 'threshold': 100, 'a': 1},
    {'type': 'opportunistic', 'mapping_type': 'spreading', 'a': 1} ]
  # eval_wmpi(rank)
  
  learn_wmpi(rank)
  