import numpy as np
import mpmath

from rvs import *
from plot_utils import *

'''
Kubernetes architecture; master assigning jobs to distributed workers.
Average cluster load = E[ro] = ar/N/Cap * E[D x S]
where
ar: Arrival rate of jobs
N: Number of workers
Cap: Capacity of each worker
k: Number of tasks in a job.
D: Total demand of a task; lifetime x resource demand
S: Slowdown experienced by each task

S is assumed to depend only on ro.
Redundancy is introduced for jobs with D < d.
'''
def E_slowdown(ar, N, Cap, k, D, S_gen, d=None, r=None):
  ## E[kD | kD <= d] = E[k x E[D | D <= d/k]]
  E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  # E_kD_given_kD_leq_d = float(
  #   mpmath.quad(lambda i: i*E_D_given_D_leq_doverk(i)*k.pdf(i), [0, mpmath.inf] ) )
  
  if d is not None:
    Pr_kD_leq_d = sum([D.cdf(d/i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    blog(Pr_kD_leq_d=Pr_kD_leq_d)
  def ro_(ro):
    S = S_gen(ro)
    if d is not None:
      ES = S.mean()
      
      ## kD <= d
      def E_cumS(k_):
        E = 0
        for i in range(1, k_+1):
          S_kplusr_i = X_n_k(S, k_+r, i)
          E += (r+1)*S_kplusr_i.mean() if i == k_ else S_kplusr_i.mean()
        return E
      EC_given_kD_leq_d = sum([E_cumS(i)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
      
      ## kD > d
      EC_given_kD_g_d = ES*sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
      
      # log(INFO, "d= {}, ro= {}".format(d, ro), EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
      EA = EC_given_kD_leq_d*Pr_kD_leq_d + \
           EC_given_kD_g_d*(1 - Pr_kD_leq_d)
    else:
      EA = k.mean()*D.mean()*S.mean()
    return ar/N/Cap * EA
  
  eq = lambda ro: ro - ro_(ro)
  l, u = 0.0001, 1
  eq_l, eq_u = eq(l), eq(u)
  print("eq_l= {}, eq_u= {}".format(eq_l, eq_u) )
  if eq_l*eq_u > 0:
    return None
  
  ro = scipy.optimize.brentq(eq, l, u)
  log(INFO, "ro= {}".format(ro), d=d)
  # for x in np.linspace(l, u, 40):
  #   print("eq({})= {}".format(x, eq(x) ) )
  
  S = S_gen(ro)
  E_S_given_kD_g_d = sum([X_n_k(S, i, i).mean()*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
  if d is not None:
    E_S_given_kD_leq_d = sum([X_n_k(S, i+r, i).mean()*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    return E_S_given_kD_leq_d*Pr_kD_leq_d + \
           E_S_given_kD_g_d*(1 - Pr_kD_leq_d)
  else:
    return E_S_given_kD_g_d
  
def arrival_rate_ub(N, Cap, k, D, S_gen):
  return N*Cap/k.mean()/D.mean()/S_gen(1).mean()  

def arrival_rate_for_load_ro(ro, N, Cap, k, D, S_gen):
  return ro*N*Cap/k.mean()/D.mean()/S_gen(ro).mean()

def plot_slowdown():
  N, Cap = 10, 100
  D = TPareto(1, 1000, 1) # Pareto(10, 2)
  k = BZipf(1, 10)
  r = 1
  
  # S_gen = lambda ro: TPareto(1, 40, 2/ro)
  def S_gen(ro):
    # a = 2 - ro # 1.5 - math.sqrt(ro)
    a = 1/ro
    return TPareto(1, 100, a)
  ar_ub = arrival_rate_ub(N, Cap, k, D, S_gen)
  print("ar_ub= {}".format(ar_ub) )
  
  # d = 2*D.mean()
  # for ar in np.linspace(ar_ub/20, ar_ub, 10):
  #   E_sl = E_slowdown(ar, N, Cap, k, D, S_gen)
  #   E_sl_wred = E_slowdown(ar, N, Cap, k, D, S_gen, d, r)
  #   print("ar= {}, E_sl= {}, E_sl_wred= {}".format(ar, E_sl, E_sl_wred) )
  
  # ar = 1/4*ar_ub # 1/2, 2/3
  ar = arrival_rate_for_load_ro(1/2, N, Cap, k, D, S_gen)
  
  l, u = 1.1*D.l_l, 0.95*D.u_l
  # for d in np.linspace(l, u, 10):
  E_sl = E_slowdown(ar, N, Cap, k, D, S_gen)
  print("E_sl= {}".format(E_sl) )
  
  d_l, E_sl_wred_l = [], []
  for d in np.logspace(math.log10(l), math.log10(u), 10):
  # for d in [l, u]:
    print("\n>> d= {}".format(d) )
    d_l.append(d)
    
    E_sl_wred = E_slowdown(ar, N, Cap, k, D, S_gen, d, r)
    blog(E_sl=E_sl, E_sl_wred=E_sl_wred)
    E_sl_wred_l.append(E_sl_wred)
    # if E_sl_wred is None:
    #   break
  
  plot.axhline(y=E_sl, label=r'w/o red', c=next(darkcolor_c) )
  plot.plot(d_l, E_sl_wred_l, label=r'w/ red', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=2)
  plot.legend()
  plot.xlabel('d', fontsize=14)
  plot.ylabel('Average slowdown', fontsize=14)
  plot.title('N= {}, Cap= {}, D$\sim {}$\n'.format(N, Cap, D.tolatex() ) + 'k$\sim${}, r= {}'.format(k, r) )
  plot.savefig('plot_slowdown.png')
  plot.gcf().clear()
  log(INFO, "done.")

if __name__ == "__main__":
  plot_slowdown()
