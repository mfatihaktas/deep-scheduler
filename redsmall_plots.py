from modeling import *
from redsmall_data import *

N, Cap = 20, 10
k = BZipf(1, 10) # BZipf(1, 5)
R = Uniform(1, 1)
b, beta_ = 10, 3 # 4
L = Pareto(b, beta_)
a, alpha_ = 1, 3
Sl = Pareto(a, alpha_)
def alpha_gen(ro):
  return alpha_

ro0_l = [0.1] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
d_l, ro0_scherid_X_l_m = get_d_l__ro0_scherid_X_l_m(alpha_)
log(INFO, "", alpha_=alpha_, d_l=d_l)

def plot_ET_wrt_d():
  r, red = 2, 'Coding'
  log(INFO, "r= {}".format(r) )
  
  dopt_l = []
  def plot_(ro0):
    log(INFO, "ro0= {}".format(ro0) )
    
    scherid_X_l_m = ro0_scherid_X_l_m[ro0]
    sim_ET_l, sim_StdT_l = [], []
    ET_wMGc_l, approx_ET_wMGc_l = [], []
    d_l_ = []
    for d in d_l:
      ET_wMGc, EW_wMGc, Prqing_wMGc = redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red)
      if ET_wMGc is None: # sys is unstable
        break
      elif ET_wMGc > 100:
        ET_wMGc = None
      ET_wMGc_l.append(ET_wMGc)
      
      d_l_.append(d)
      X_l_m = scherid_X_l_m['d={}'.format(d) ]
      sim_ET_l.append(np.mean(X_l_m['ET_l'] ) )
      sim_StdT_l.append(np.std(X_l_m['ET_l'] ) )
      
      approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = redsmall_approx_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red)
      if approx_ET_wMGc > 100:
        approx_ET_wMGc = None
      approx_ET_wMGc_l.append(approx_ET_wMGc)
    plot.errorbar(d_l_, sim_ET_l, yerr=sim_StdT_l, label='Simulation', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.plot(d_l_, ET_wMGc_l, label='M/G/c', c=NICE_BLUE, marker='o', ls=':')
    plot.plot(d_l_, approx_ET_wMGc_l, label='Asymptotic', c=NICE_GREEN, marker='p', ls=':', mew=0.5, ms=8)
    
    # d_opt = optimal_d_pareto(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, red, max_d=max(d_l_) )
    # dopt_l.append(d_opt)
    # # d_opt = min(d_opt, max(d_l_) )
    # # if d_opt <= max(d_l_):
    # ET_wMGc, EW_wMGc, Prqing_wMGc = ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, d_opt, red)
    # log(INFO, "*** ro0= {}, d_opt= {}, ET_wMGc= {}".format(ro0, d_opt, ET_wMGc) )
    # plot.plot([d_opt], [ET_wMGc], label=r'$\sim$optimal', c='orangered', marker='x', ls=':', mew=3, ms=10)
    
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
  
  for ro0 in ro0_l:
    plot_(ro0)
  # plot_(ro0=0.1)
  
  log(INFO, "done;", ro0_l=ro0_l, dopt_l=dopt_l)

def plot_ESl_ET_vs_ro__redsmall_vs_drl():
  ro0_scherid_X_l_m = get_data_redsmall_vs_drl(alpha_)
  
  def profile(ro, scherid, X, ulim=float('Inf') ):
    l = ro0_scherid_X_l_m[ro][scherid][X]
    m, s = np.mean(l), np.std(l)
    if m > ulim:
      m, s = float('NaN'), float('NaN')
    return m, s
  
  RLScher_ESl_l, Redsmall_ESl_l = [], []
  RLScher_ESl_err_l, Redsmall_ESl_err_l = [], []
  RLScher_ET_l, Redsmall_ET_l = [], []
  RLScher_ET_err_l, Redsmall_ET_err_l = [], []
  for ro in ro0_l:
    mean, stdev = profile(ro, 'RLScher', 'ESl_l')
    RLScher_ESl_l.append(mean)
    RLScher_ESl_err_l.append(stdev)
    mean, stdev = profile(ro, 'Redsmall', 'ESl_l')
    Redsmall_ESl_l.append(mean)
    Redsmall_ESl_err_l.append(stdev)
    
    mean, stdev = profile(ro, 'RLScher', 'ET_l')
    RLScher_ET_l.append(mean)
    RLScher_ET_err_l.append(stdev)
    mean, stdev = profile(ro, 'Redsmall', 'ET_l')
    Redsmall_ET_l.append(mean)
    Redsmall_ET_err_l.append(stdev)
  
  ## ESl
  plot.errorbar(ro0_l, RLScher_ESl_l, yerr=RLScher_ESl_err_l, label='Redundant-RL', c=NICE_RED, marker=next(marker_c), linestyle=':', mew=0.5, ms=8)
  plot.errorbar(ro0_l, Redsmall_ESl_l, yerr=Redsmall_ESl_err_l, label='Redundant-small', c=NICE_GREEN, marker=next(marker_c), linestyle=':', mew=0.5, ms=8)
  
  fontsize = 18
  prettify(plot.gca() )
  plot.legend(framealpha=0.5, loc='best', numpoints=1)
  plot.xticks(rotation=70)
  # plot.yscale('log')
  plot.xlabel(r'Baseline offered load $\rho_0$', fontsize=fontsize)
  plot.ylabel('Average job slowdown', fontsize=fontsize)
  # plot.title(r'$\rho= {}$'.format(ro), fontsize=fontsize)
  plot.gcf().set_size_inches(4, 4)
  plot.savefig('plot_ESl_vs_ro__redsmall_vs_drl.png', bbox_inches='tight')
  plot.gcf().clear()
  
  ## ET
  plot.errorbar(ro0_l, RLScher_ET_l, yerr=RLScher_ET_err_l, label='Redundant-RL', c=NICE_RED, marker=next(marker_c), linestyle=':', mew=0.5, ms=8)
  plot.errorbar(ro0_l, Redsmall_ET_l, yerr=Redsmall_ET_err_l, label='Redundant-small', c=NICE_GREEN, marker=next(marker_c), linestyle=':', mew=0.5, ms=8)
  
  prettify(plot.gca() )
  plot.legend(framealpha=0.5, loc='best', numpoints=1)
  plot.xticks(rotation=70)
  plot.xlabel(r'Baseline offered load $\rho_0$', fontsize=fontsize)
  plot.ylabel('Average job completion time', fontsize=fontsize)
  plot.gcf().set_size_inches(4, 4)
  plot.savefig('plot_ET_vs_ro__redsmall_vs_drl.png', bbox_inches='tight')
  plot.gcf().clear()
  
  log(INFO, "done.")

if __name__ == "__main__":
  # ar = round(ar_for_ro(ro, N, Cap, k, R, L, S), 2)
  # sinfo_m = {
  #   'njob': 5000*N,  #   'nworker': N, 'wcap': Cap, 'ar': ar,  #   'k_rv': k,  #   'reqed_rv': R,  #   'lifetime_rv': L,  #   'straggle_m': {'slowdown': lambda load: S.sample() } }
  # mapping_m = {'type': 'spreading'}
  # sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  
  # u = 40*L.mean()*Sl.mean()
  # for d in [0, *np.logspace(math.log10(l), math.log10(u), 20) ]:
  
  plot_ET_wrt_d()
  # plot_ESl_ET_vs_ro__redsmall_vs_drl()
