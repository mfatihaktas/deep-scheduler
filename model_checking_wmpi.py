import sys
from mpi4py import MPI

from model_checking import *

def sim_ET_EW_Prqing(d):
  sching_m['threshold'] = d
  sim_m = sim(sinfo_m, mapping_m, sching_m, "N{}_C{}".format(N, Cap) )
  # blog(sim_m=sim_m)
  return sim_m['responsetime_mean'], sim_m['waittime_mean'], sim_m['frac_jobs_waited_inq']

def plot_ET_wrt_d(rank):
  log(INFO, "rank= {}, starting;".format(rank) )
  sys.stdout.flush()
  
  if rank == 0:
    l = L.l_l*S.l_l
    u = 40*L.mean()*S.mean()
    d_l = []
    sim_Prqing_l, Prqing_wMGc_l, approx_Prqing_wMGc_l = [], [], []
    sim_ET_l, ET_wMGc_l, approx_ET_wMGc_l = [], [], []
    for d in [0, *np.logspace(math.log10(l), math.log10(u), 20) ]:
      print(">> d= {}".format(d) )
      sys.stdout.flush()
      for prank in range(1, num_mpiprocs):
        d_ = np.array([d], dtype='i')
        comm.Send([d_, MPI.INT], dest=prank)
      
      sET, sEW, sPrqing = sim_ET_EW_Prqing(d)
      print("rank= {}, sET= {}, sEW= {}, sPrqing= {}".format(rank, sET, sEW, sPrqing) )
      sET_l, sEW_l, sPrqing_l = [sET], [sEW], [sPrqing]
      for prank in range(1, num_mpiprocs):
        sET_sEW_sPrqing = np.empty(3, dtype=np.float64)
        comm.Recv([sET_sEW_sPrqing, MPI.FLOAT], source=prank)
        sET_l.append(sET_sEW_sPrqing[0] )
        sEW_l.append(sET_sEW_sPrqing[1] )
        sPrqing_l.append(sET_sEW_sPrqing[2] )
      sim_ET, sim_EW, sim_Prqing = np.mean(sET_l), np.mean(sEW_l), np.mean(sPrqing_l)
      if d == 0:
        sim_ET0 = sim_ET
      print("*** sim_ET= {}, sim_EW= {}, sim_Prqing= {}".format(sim_ET, sim_EW, sim_Prqing) )
      blog(sET_l=sET_l, sEW_l=sEW_l, sPrqing_l=sPrqing_l)
      
      ET_wMGc, EW_wMGc, Prqing_wMGc = ET_EW_Prqing_pareto_wMGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      print("*** ET_wMGc= {}, EW_wMGc= {}, Prqing_wMGc= {}".format(ET_wMGc, EW_wMGc, Prqing_wMGc) )
      approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = approx_ET_EW_Prqing_pareto_wMGc(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      print("*** approx_ET_wMGc= {}, approx_EW_wMGc= {}, approx_Prqing_wMGc= {}".format(approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc) )
      sys.stdout.flush()
      
      d_l.append(d)
      sim_Prqing_l.append(sim_Prqing)
      Prqing_wMGc_l.append(Prqing_wMGc)
      approx_Prqing_wMGc_l.append(approx_Prqing_wMGc)
      sim_ET_l.append(sim_ET)
      ET_wMGc_l.append(ET_wMGc)
      approx_ET_wMGc_l.append(approx_ET_wMGc)
      
      if sim_ET > 2*sim_ET0:
        break
    for prank in range(1, num_mpiprocs):
      d = np.array([-1], dtype='i')
      comm.Send([d, MPI.INT], dest=prank)
    blog(sim_ET_l=sim_ET_l, ET_wMGc_l=ET_wMGc_l, approx_ET_wMGc_l=approx_ET_wMGc_l)
    
    fig, axs = plot.subplots(1, 2)
    fontsize = 14
    ax = axs[0]
    plot.sca(ax)
    plot.plot(d_l, sim_ET_l, label='Sim', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    plot.plot(d_l, ET_wMGc_l, label='M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    plot.plot(d_l, approx_ET_wMGc_l, label='Approx M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    prettify(ax)
    plot.legend(loc='best', framealpha=0.5)
    plot.xscale('log')
    plot.xlabel('d', fontsize=fontsize)
    plot.ylabel('E[T]', fontsize=fontsize)
    # 
    ax = axs[1]
    plot.sca(ax)
    plot.plot(d_l, sim_Prqing_l, label='Sim', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    plot.plot(d_l, Prqing_wMGc_l, label='M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    plot.plot(d_l, approx_Prqing_wMGc_l, label='Approx M/G/c model', c=next(darkcolor_c), marker=next(marker_c), ls=':')
    prettify(ax)
    plot.legend(loc='best', framealpha=0.5)
    plot.xscale('log')
    plot.xlabel('d', fontsize=fontsize)
    plot.ylabel('Pr\{Queueing\}', fontsize=fontsize)
    
    plot.subplots_adjust(hspace=1)
    st = plot.suptitle(r'$N= {}$, $C= {}$, $\rho_0= {}$, $r= {}$, $k \sim$ {}'.format(N, Cap, ro, r, k) + '\n' + r'$R \sim$ {}, $L \sim$ {}, $S \sim$ {}'.format(R, L, S) )
    fig.set_size_inches(2*5, 5)
    plot.savefig('plot_ET_wrt_d.png', bbox_extra_artists=(st,), bbox_inches='tight')
    fig.clear()
  else:
    while True:
      d = np.empty(1, dtype='i')
      comm.Recv([d, MPI.INT], source=0)
      d = d[0]
      if d == -1:
        break
      
      print("rank= {}, will sim for d= {}".format(rank, d) )
      sys.stdout.flush()
      sET, sEW, sPrqing = sim_ET_EW_Prqing(d)
      print("rank= {}, sET= {}, sEW= {}, sPrqing= {}".format(rank, sET, sEW, sPrqing) )
      sys.stdout.flush()
      sET_sEW_sPrqing = np.array([sET, sEW, sPrqing], dtype=np.float64)
      comm.Send([sET_sEW_sPrqing, MPI.FLOAT], dest=0)
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
    'njob': 10*N, # 5000*N,
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
