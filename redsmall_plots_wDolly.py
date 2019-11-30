from modeling_wDolly import *
from redsmall_data_wDolly import *

N, Cap = 20, 10
k = BZipf(1, 10) # BZipf(1, 5)
b, beta_ = 10, 3
L = Pareto(b, beta_)
Sl = Dolly()
r, red = 2, 'Coding'

ro0_l = [0.9] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] # , 0.8, 0.9]
d_l, ro0_scherid_X_l_m = get_d_l__ro0_scherid_X_l_m(beta_)
# log(INFO, "", d_l=d_l, ro0_scherid_X_l_m=ro0_scherid_X_l_m)

def plot_ET_wrt_d():
  log(INFO, "r= {}".format(r) )
  
  dopt_l = []
  def plot_(ro0):
    log(INFO, "ro0= {}".format(ro0) )
    
    scherid_X_l_m = ro0_scherid_X_l_m[ro0]
    sim_ET_l, sim_StdT_l = [], []
    ET_wMGc_l, approx_ET_wMGc_l = [], []
    d_l_ = []
    for d in d_l:
    # for d in sorted([float(k[2:]) for k, _ in scherid_X_l_m.items() ] ):
    # for key in [k for k, _ in scherid_X_l_m.items() ]:
      X_l_m = scherid_X_l_m['d={}'.format(d) ]
      # X_l_m = scherid_X_l_m[key]
      # d = float(key[2:])
      sim_ET = np.mean(X_l_m['ET_l'] )
      if sim_ET > 300:
        break
      d_l_.append(d)
      sim_ET_l.append(sim_ET)
      sim_StdT_l.append(np.std(X_l_m['ET_l'] ) )
      
      ET_wMGc, EW_wMGc, Prqing_wMGc = redsmall_ET_EW_Prqing_wMGc_wSl(ro0, N, Cap, k, r, L, Sl, d)
      if ET_wMGc is None or ET_wMGc > 300:
        ET_wMGc = None
      ET_wMGc_l.append(ET_wMGc)
      
      approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = redsmall_approx_ET_EW_Prqing_wMGc_wSl(ro0, N, Cap, k, r, L, Sl, d)
      if approx_ET_wMGc is None or approx_ET_wMGc > 300:
        approx_ET_wMGc = None
      approx_ET_wMGc_l.append(approx_ET_wMGc)
      
      log(INFO, "\n*** d= {}".format(d), sim_ET=sim_ET, ET_wMGc=ET_wMGc, approx_ET_wMGc=approx_ET_wMGc)
    
    plot.errorbar(d_l_, sim_ET_l, yerr=sim_StdT_l, label='Simulation', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.plot(d_l_, ET_wMGc_l, label='M/G/c', c=NICE_BLUE, marker='o', ls=':')
    plot.plot(d_l_, approx_ET_wMGc_l, label='Asymptotic', c=NICE_GREEN, marker='p', ls=':', mew=0.5, ms=8)
    
    # d_opt = redsmall_optimal_d(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, red, max_d=max(d_l_) )
    # dopt_l.append(d_opt)
    # # d_opt = min(d_opt, max(d_l_) )
    # # if d_opt <= max(d_l_):
    # ET_wMGc, EW_wMGc, Prqing_wMGc = redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta_, a, alpha_gen, d_opt, red)
    # log(INFO, "*** ro0= {}, d_opt= {}, max_d= {}, ET_wMGc= {}".format(ro0, d_opt, max(d_l_), ET_wMGc) )
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
    plot.savefig('plot_ET_wrt_d_ro{}_wDolly.png'.format(ro0), bbox_inches='tight')
    fig.clear()
  
  for ro0 in ro0_l:
    plot_(ro0)
  # plot_(ro0=0.1)
  
  log(INFO, "done;", ro0_l=ro0_l, dopt_l=dopt_l)

if __name__ == "__main__":
  plot_ET_wrt_d()
