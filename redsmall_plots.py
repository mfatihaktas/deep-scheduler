from modeling import *
from plot_data import *

# k = BZipf(1, 5)
# alpha = 4
k = BZipf(1, 10)
alpha = 3

ro_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

d_l, ro_scherid_X_l_m = get_d_l__ro_scherid_X_l_m(alpha, k.u_l)

def plot_ET_wrt_d():
  r, red = 2, 'Coding'
  log(INFO, "r= {}".format(r) )
  
  dopt_l = []
  def plot_(ro0):
    log(INFO, "ro0= {}".format(ro0) )
    
    scherid_X_l_m = ro_scherid_X_l_m[ro0]
    sim_ET_l, sim_StdT_l = [], []
    ET_wMGc_l, approx_ET_wMGc_l = [], []
    d_l_ = []
    for d in d_l:
      ET_wMGc, EW_wMGc, Prqing_wMGc = ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      if ET_wMGc is None: # sys is unstable
        break
      ET_wMGc_l.append(ET_wMGc)
      
      d_l_.append(d)
      X_l_m = scherid_X_l_m['d={}'.format(d) ]
      sim_ET_l.append(np.mean(X_l_m['ET_l'] ) )
      sim_StdT_l.append(np.std(X_l_m['ET_l'] ) )
      
      approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = approx_ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      approx_ET_wMGc_l.append(approx_ET_wMGc)
    plot.errorbar(d_l_, sim_ET_l, yerr=sim_StdT_l, label='Simulation', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.plot(d_l_, ET_wMGc_l, label='M/G/c', c=NICE_BLUE, marker='o', ls=':')
    plot.plot(d_l_, approx_ET_wMGc_l, label='Asymptotic', c=NICE_GREEN, marker='p', ls=':', mew=0.5, ms=8)
    
    d_opt = optimal_d_pareto(ro0, N, Cap, k, r, b, beta, a, alpha_gen, red, max_d=max(d_l_) )
    dopt_l.append(d_opt)
    # d_opt = min(d_opt, max(d_l_) )
    # if d_opt <= max(d_l_):
    ET_wMGc, EW_wMGc, Prqing_wMGc = ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d_opt, red)
    log(INFO, "*** ro0= {}, d_opt= {}, ET_wMGc= {}".format(ro0, d_opt, ET_wMGc) )
    plot.plot([d_opt], [ET_wMGc], label=r'$\sim$optimal', c='orangered', marker='x', ls=':', mew=3, ms=10)
    
    plot.xscale('log')
    plot.xlim(right=max(d_l_)*2)
    prettify(plot.gca() )
    fontsize = 21
    plot.legend(loc='best', framealpha=0.5, fontsize=14, numpoints=1)
    plot.xlabel(r'$d$', fontsize=fontsize)
    plot.ylabel(r'$E[T]$', fontsize=fontsize)
    
    # plot.title(r'$N= {}$, $Cap= {}$, $\rho_0= {}$, $r= {}$'.format(N, Cap, ro, r) + '\n' \
    #   + r'$k \sim${}, $L \sim${}, $Sl \sim${}'.format(k.to_latex(), L.to_latex(), Sl.to_latex() ) )
    # plot.gca().title.set_position([.5, 1.05] )
    plot.title(r'$\rho_0= {}$'.format(ro0), fontsize=fontsize)
    fig = plot.gcf()
    fig.set_size_inches(4, 4)
    plot.savefig('plot_ET_wrt_d_ro{}.png'.format(ro0), bbox_inches='tight')
    fig.clear()
  
  for ro0 in ro_l:
    plot_(ro0)
  # plot_(ro0=0.1)
  
  log(INFO, "done;", ro_l=ro_l, dopt_l=dopt_l)

def plot_opt_redsmall_vs_drl():
  get_data_redsmall_vs_drl(alpha, k_max)
  d_l, ro_scherid_X_l_m = get_d_l__ro_scherid_X_l_m(alpha, k.u_l)


if __name__ == "__main__":
  N, Cap = 20, 10
  k = BZipf(1, 10) # BZipf(1, 5)
  R = Uniform(1, 1)
  b, beta = 10, 3 # 4
  L = Pareto(b, beta)
  a, alpha = 1, 3
  Sl = Pareto(a, alpha)
  def alpha_gen(ro):
    return alpha
  
  # ar = round(ar_for_ro(ro, N, Cap, k, R, L, S), 2)
  # sinfo_m = {
  #   'njob': 5000*N,  #   'nworker': N, 'wcap': Cap, 'ar': ar,  #   'k_rv': k,  #   'reqed_rv': R,  #   'lifetime_rv': L,  #   'straggle_m': {'slowdown': lambda load: S.sample() } }
  # mapping_m = {'type': 'spreading'}
  # sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  
  # u = 40*L.mean()*Sl.mean()
  # for d in [0, *np.logspace(math.log10(l), math.log10(u), 20) ]:
  
  plot_ET_wrt_d()
