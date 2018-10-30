import numpy as np
import mpmath

from rvs import *

'''
Kubernetes architecture; master assigning jobs to distributed workers.
Average cluster load = E[ro] = ar/N/Cap * E[D x S]
where
ar: Arrival rate of jobs
N: Number of workers
Cap: Capacity of each worker
D: Total demand of a job; lifetime x resource demand
S: Slowdown experienced by each task

S is assumed to depend only on ro.
Redundancy is introduced for jobs with D < D0.
'''
def E_slowdown(ar, N, Cap, D, S_gen, D0=None):
  def ro_(ro):
    if D0 is not None:
      ED_given_D_leq_D0 = mean(D, given_X_leq_x=True, x=D0)
      S = S_gen(ro)
      S_2_1 = X_n_k(S, 2, 1)
      ES_given_D_leq_D0 = S_2_1.moment(1)
      
      ED_given_D_g_D0 = mean(D, given_X_leq_x=False, x=D0)
      ES_given_D_g_D0 = S.moment(1)
      # blog(E_S_2_1=ES_given_D_leq_D0, ES=ES_given_D_g_D0)
      
      Pr_D_leq_D0 = D.cdf(D0)
      
      EA = 2*ED_given_D_leq_D0*ES_given_D_leq_D0*Pr_D_leq_D0 + \
           ED_given_D_g_D0*ES_given_D_g_D0*(1 - Pr_D_leq_D0)
    else:
      EA = D.mean()*S_gen(ro).mean()
    return ar/N/Cap * EA
  
  eq = lambda ro: ro - ro_(ro)
  l, u = 0.0001, 1
  roots = scipy.optimize.brentq(eq, l, u)
  print("roots= {}".format(roots) )
  # for x in np.linspace(l, u, 40):
  #   print("eq({})= {}".format(x, eq(x) ) )
  
  S = S_gen(roots)
  return S.mean()

def arrival_rate_ub(N, Cap, D, S_gen):
  return N*Cap/D.mean()/S_gen(1).mean()  

def plot_slowdown():
  N, Cap = 10, 100
  D = TPareto(1, 10000, 2) # Pareto(10, 2)
  
  # S_gen = lambda ro: TPareto(1, 40, 2/ro)
  def S_gen(ro):
    a = 1.5 - math.sqrt(ro) # 2 - ro
    return TPareto(1, 1000, a)
  ar_ub = arrival_rate_ub(N, Cap, D, S_gen)
  print("ar_ub= {}".format(ar_ub) )
  
  # D0 = 2*D.mean()
  # for ar in np.linspace(ar_ub/20, ar_ub, 10):
  #   E_sl = E_slowdown(ar, N, Cap, D, S_gen)
  #   E_sl_wred = E_slowdown(ar, N, Cap, D, S_gen, D0)
  #   print("ar= {}, E_sl= {}, E_sl_wred= {}".format(ar, E_sl, E_sl_wred) )
  
  ar = 1/2*ar_ub # 2/3
  l, u = 1.1*D.l_l, 0.95*D.u_l
  # for D0 in np.linspace(l, u, 10):
  for D0 in np.logspace(math.log10(l), math.log10(u), 10):
    print("D0= {}".format(D0) )
    E_sl = E_slowdown(ar, N, Cap, D, S_gen)
    E_sl_wred = E_slowdown(ar, N, Cap, D, S_gen, D0)
    blog(E_sl=E_sl, E_sl_wred=E_sl_wred)
  

if __name__ == "__main__":
  plot_slowdown()
