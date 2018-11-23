import sys
from mpi4py import MPI

from model_checking import *

def sim_ET_EW(d):
  sching_m['threshold'] = d
  sim_m = sim(sinfo_m, mapping_m, sching_m, "N{}_C{}".format(N, Cap) )
  # blog(sim_m=sim_m)
  return sim_m['responsetime_mean'], sim_m['waittime_mean']

def plot_ET_wrt_d(rank):
  log(INFO, "rank= {}, starting;".format(rank) )
  sys.stdout.flush()
  
  if rank == 0:
    l = L.l_l*S.l_l
    u = 40*L.mean()*S.mean()
    d_l, sim_ET_l, ET_w_MGc_l, approx_ET_w_MGc_l = [], [], [], []
    for d in [0, *np.logspace(math.log10(l), math.log10(u), 20) ]:
      print(">> d= {}".format(d) )
      sys.stdout.flush()
      for prank in range(1, num_mpiprocs):
        d_ = np.array([d], dtype='i')
        comm.Send([d_, MPI.INT], dest=prank)
      
      sET, sEW = sim_ET_EW(d)
      print("rank= {}, sET= {}, sEW= {}".format(rank, sET, sEW) )
      sET_l, sEW_l = [sET], [sEW]
      for prank in range(1, num_mpiprocs):
        sET_sEW = np.empty(2, dtype=np.float64)
        comm.Recv([sET_sEW, MPI.FLOAT], source=prank)
        sET_l.append(sET_sEW[0] )
        sEW_l.append(sET_sEW[1] )
      sim_ET = np.mean(sET_l)
      sim_EW = np.mean(sEW_l)
      if d == 0:
        sim_ET0 = sim_ET
      print("*** sim_ET= {}, sim_EW= {}".format(sim_ET, sim_EW) )
      blog(sET_l=sET_l, sEW_l=sEW_l)
      ET_w_MGc, EW_w_MGc = ET_EW_pareto_w_MGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      print("*** ET_w_MGc= {}, EW_w_MGc= {}".format(ET_w_MGc, EW_w_MGc) )
      approx_ET_w_MGc, approx_EW_w_MGc = approx_ET_EW_pareto_w_MGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      print("*** approx_ET_w_MGc= {}, approx_EW_w_MGc= {}".format(approx_ET_w_MGc, approx_EW_w_MGc) )
      sys.stdout.flush()
      
      d_l.append(d)
      sim_ET_l.append(sim_ET)
      ET_w_MGc_l.append(ET_w_MGc)
      approx_ET_w_MGc_l.append(approx_ET_w_MGc)
      
      if sim_ET > 2*sim_ET0:
        break
    for prank in range(1, num_mpiprocs):
      d = np.array([-1], dtype='i')
      comm.Send([d, MPI.INT], dest=prank)
    
    blog(sim_ET_l=sim_ET_l, ET_w_MGc_l=ET_w_MGc_l, approx_ET_w_MGc_l=approx_ET_w_MGc_l)
    plot.plot(d_l, sim_ET_l, label='Sim', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, ET_w_MGc_l, label='M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, approx_ET_w_MGc_l, label='Approx M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    # plot.plot(d_l, ET_l, label='Heavy-tail model', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    prettify(plot.gca() )
    plot.legend(loc='best', framealpha=0.5)
    plot.xscale('log')
    fontsize = 14
    plot.xlabel('d', fontsize=fontsize)
    plot.ylabel('E[T]', fontsize=fontsize)
    plot.title(r'$N= {}$, $C= {}$, $\rho_0= {}$, $r= {}$, $k \sim$ {}'.format(N, Cap, ro, r, k) + '\n' + r'$R \sim$ {}, $L \sim$ {}, $S \sim$ {}'.format(R, L, S) )
    plot.gcf().set_size_inches(5, 5)
    plot.savefig('plot_ET_wrt_d.png', bbox_inches='tight')
    plot.gcf().clear()
  else:
    while True:
      d = np.empty(1, dtype='i')
      comm.Recv([d, MPI.INT], source=0)
      d = d[0]
      if d == -1:
        break
      
      print("rank= {}, will sim for d= {}".format(rank, d) )
      sys.stdout.flush()
      sET, sEW = sim_ET_EW(d)
      print("rank= {}, sET= {}, sEW= {}".format(rank, sET, sEW) )
      sys.stdout.flush()
      sET_sEW = np.array([sET, sEW], dtype=np.float64)
      comm.Send([sET_sEW, MPI.FLOAT], dest=0)
  log(INFO, "done.")
  sys.stdout.flush()

if __name__ == "__main__":
  comm = MPI.COMM_WORLD
  num_mpiprocs = comm.Get_size()
  rank = comm.Get_rank()
  
  N, Cap = 20, 10
  k = BZipf(1, 5) # DUniform(1, 1)
  R = Uniform(1, 1)
  b, beta = 10, 4
  L = Pareto(b, beta) # TPareto(10, 10**6, 4)
  a, alpha = 1, 3 # 1, 4
  S = Pareto(a, alpha) # Uniform(1, 1)
  def alpha_gen(ro):
    return alpha
  ro = 0.6
  red, r = 'Coding', 2
  print("ro= {}".format(ro) )
  
  ar = round(ar_for_ro(ro, N, Cap, k, R, L, S), 2)
  sinfo_m = {
    'njob': 5000*N,
    'nworker': N, 'wcap': Cap, 'ar': ar,
    'k_rv': k,
    'reqed_rv': R,
    'lifetime_rv': L,
    'straggle_m': {'slowdown': lambda load: S.sample() } }
  mapping_m = {'type': 'spreading'}
  sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  log(INFO, "rank= {}, num_mpiprocs= {}".format(rank, num_mpiprocs) , sinfo_m=sinfo_m, sching_m=sching_m, mapping_m=mapping_m)
  sys.stdout.flush()
  
  plot_ET_wrt_d(rank)
