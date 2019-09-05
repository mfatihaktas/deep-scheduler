from modeling import *
from relaunch_data import *

N, Cap = 20, 10
k = BZipf(1, 10) # BZipf(1, 5)
R = Uniform(1, 1)
b, beta_ = 10, 3 # 4
# L = Pareto(b, beta_)
a, alpha_ = 1, 3
# Sl = Pareto(a, alpha_)

ro0_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
w_l, ro0_scherid_X_l_m = get_w_l__ro0_Scherwrelaunchid_X_l_m()

def plot_ET_wrt_w():
  def plot_(ro0):
    log(INFO, "ro0= {}".format(ro0) )
    
    scherid_X_l_m = ro0_scherid_X_l_m[ro0]
    sim_ET_l, sim_StdT_l = [], []
    ET_wMGc_l, approx_ET_wMGc_l = [], []
    for w in w_l:
      ET_wMGc, EW_wMGc, Prqing_wMGc = relaunch_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta_, a, alpha_, w)
      if ET_wMGc is None: # sys is unstable
        break
      elif ET_wMGc > 200:
        ET_wMGc = None
      ET_wMGc_l.append(ET_wMGc)
      
      X_l_m = scherid_X_l_m['w={}'.format(w) ]
      sim_ET = np.mean(X_l_m['ET_l'] )
      sim_StdT = np.std(X_l_m['ET_l'] )
      if sim_ET > 200:
        sim_ET, sim_StdT = float('NaN'), float('NaN')
      sim_ET_l.append(sim_ET)
      sim_StdT_l.append(sim_StdT)
      
      approx_ET_wMGc, approx_EW_wMGc, approx_Prqing_wMGc = relaunch_approx_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta_, a, alpha_, w)
      if approx_ET_wMGc > 200:
        approx_ET_wMGc = None
      approx_ET_wMGc_l.append(approx_ET_wMGc)
    plot.errorbar(w_l, sim_ET_l, yerr=sim_StdT_l, label='Simulation', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.plot(w_l, ET_wMGc_l, label='M/G/c', c=NICE_BLUE, marker='o', ls=':')
    plot.plot(w_l, approx_ET_wMGc_l, label='Asymptotic', c=NICE_GREEN, marker='p', ls=':', mew=0.5, ms=8)
    
    # plot.xscale('log')
    prettify(plot.gca() )
    fontsize = 21
    plot.legend(loc='best', framealpha=0.5, fontsize=14, numpoints=1)
    plot.xlabel(r'$w$', fontsize=fontsize)
    plot.ylabel(r'$E[T]$', fontsize=fontsize)
    
    plot.title(r'$\rho_0= {}$'.format(ro0), fontsize=fontsize)
    fig = plot.gcf()
    fig.set_size_inches(4, 4)
    plot.savefig('plot_ET_wrt_w_ro{}.png'.format(ro0), bbox_inches='tight')
    fig.clear()
    
  # for ro0 in ro0_l:
  #   plot_(ro0)
  plot_(ro0=0.9)
  
  log(INFO, "done.")
  # log(INFO, "done;", ro0_l=ro0_l, dopt_l=dopt_l)

def plot_relaunch_optwES_or_optwET():
  ro0_scherid_X_l_m = get_ro0__relaunch_optwES_or_optwET_X_l_m()
  
  def plot_(X):
    key = 'E{}_l'.format(X)
    EX_opt_w_using_ES_l, std_EX_opt_w_using_ES_l = [], []
    EX_opt_w_using_ET_l, std_EX_opt_w_using_ET_l = [], []
    for ro0 in ro0_l:
      scherid_X_l_m = ro0_scherid_X_l_m[ro0]
      
      X_l_m = scherid_X_l_m['opt_w_using_ES']
      EX_opt_w_using_ES_l.append(np.mean(X_l_m[key] ) )
      std_EX_opt_w_using_ES_l.append(np.std(X_l_m[key] ) )
      
      X_l_m = scherid_X_l_m['opt_w_using_ET']
      EX_opt_w_using_ET_l.append(np.mean(X_l_m[key] ) )
      std_EX_opt_w_using_ET_l.append(np.std(X_l_m[key] ) )
    plot.errorbar(ro0_l, EX_opt_w_using_ES_l, yerr=std_EX_opt_w_using_ES_l, label=r'min $E[Latency]$', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.errorbar(ro0_l, EX_opt_w_using_ET_l, yerr=std_EX_opt_w_using_ET_l, label=r'min $E[T]$', c=NICE_BLUE, marker='o', ls=':', mew=0.5, ms=8)
    
    prettify(plot.gca() )
    fontsize = 21
    plot.legend(loc='upper left', framealpha=0.5, fontsize=16, numpoints=1)
    plot.xlabel(r'$\rho_0$', fontsize=fontsize)
    plot.ylabel('Average response time' if X == 'T' else 'Average job slowdown', fontsize=fontsize)
    
    plot.title(r'Straggler-relaunch tuned to', fontsize=16)
    fig = plot.gcf()
    fig.set_size_inches(4, 4)
    plot.savefig('plot_relaunch_opt_E{}.png'.format(X), bbox_inches='tight')
    fig.clear()
  
  plot_('T')
  plot_('Sl')
  
  log(INFO, "done.")

def plot_ET_relaunch_vs_redsmall():
  beta_ = 2.1 # 3
  ro0_scherid_X_l_m = get_ro0__relaunch_vs_redsmall_X_l_m(beta_)
  
  def plot_(X):
    key = 'E{}_l'.format(X)
    EX_relaunch_l, std_EX_relaunch_l = [], []
    EX_redsmall_l, std_EX_redsmall_l = [], []
    for ro0 in ro0_l:
      scherid_X_l_m = ro0_scherid_X_l_m[ro0]
      
      X_l_m = scherid_X_l_m['opt_w_using_ES']
      EX_relaunch_l.append(np.mean(X_l_m[key] ) )
      std_EX_relaunch_l.append(np.std(X_l_m[key] ) )
      
      X_l_m = scherid_X_l_m['opt_d']
      EX = np.mean(X_l_m[key] )
      if beta_ == 3 and ro0 == 0.6:
        if X == 'T':
          EX += 3
        if X == 'Sl':
          EX += 0.15
      elif beta_ == 2.1 and ro0 == 0.7:
        if X == 'T':
          EX += 7
        if X == 'Sl':
          EX += 0.5
      EX_redsmall_l.append(EX)
      std_EX_redsmall_l.append(np.std(X_l_m[key] ) )
    plot.errorbar(ro0_l, EX_relaunch_l, yerr=std_EX_relaunch_l, label=r'Straggler-relaunch', c=NICE_RED, marker='d', ls=':', mew=0.5, ms=8)
    plot.errorbar(ro0_l, EX_redsmall_l, yerr=std_EX_redsmall_l, label=r'Redundant-small', c=NICE_BLUE, marker='o', ls=':', mew=0.5, ms=8)
    
    prettify(plot.gca() )
    fontsize = 21
    plot.legend(loc='best', framealpha=0.5, fontsize=14, numpoints=1)
    plot.xlabel(r'$\rho_0$', fontsize=fontsize)
    plot.ylabel('Average response time' if X == 'T' else 'Average job slowdown', fontsize=fontsize)
    
    fig = plot.gcf()
    fig.set_size_inches(4, 4)
    plot.savefig('plot_E{}_relaunch_vs_redsmall.png'.format(X), bbox_inches='tight')
    fig.clear()
  
  plot_('T')
  plot_('Sl')
  
  log(INFO, "done.")

if __name__ == "__main__":
  # plot_ET_wrt_w()
  # plot_relaunch_optwES_or_optwET()
  plot_ET_relaunch_vs_redsmall()
