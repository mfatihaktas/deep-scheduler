import numpy as np
import mpmath, scipy

from rvs import *
from plot_utils import *

def G(z):
  return scipy.special.gamma(z)

def I(u_l, m, n):
  # den = B(m, n)
  # if den == 0:
  #   return None
  # return B(m, n, u_l=u_l)/den
  return scipy.special.betainc(m, n, u_l)

def B(m, n, u_l=1):
  # return mpmath.quad(lambda x: x**(m-1) * (1-x)**(n-1), [0.0001, u_l] )
  func = lambda x: x**(m-1) * (1-x)**(n-1)
  result, abserr = scipy.integrate.quad(func, 0.0001, u_l)
  return result # round(result, 2)
  # if u_l == 1:
  #   return scipy.special.beta(m, n)
  # else:
  #   return I(u_l, m, n)*B(m, n)

def EC_k_n_pareto(k, n, loc, a):
  if n > 170:
    return loc/(a-1) * (a*n - (n-k)*((n+1)/(n-k+1))**(1/a) )
  return loc*n/(a-1) * (a - G(n)/G(n-k)*G(n-k+1-1/a)/G(n+1-1/a) )

def compare_exact_approx():
  # N, Cap = 10, 100
  k = BZipf(1, 10)
  r = 1.5
  b, beta = 10, 1.1
  D = Pareto(b, beta)
  a, alpha = 1, 2
  S = Pareto(a, alpha)
  
  def exact(d):
    E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
    E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
    
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, int(i*r), a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    return EC_given_kD_leq_d
  
  def approx(d):
    # print("B(1-beta, 1, d/k)= {}".format(B(1-beta, 1, d/10) ) )
    def E_D_given_D_leq_doverk(k):
      if b >= d/k:
        return 0
      else:
        # return (b + b**beta*(B(1-beta, 1, d/k) - B(1-beta, 1, b) ) )/(1 - b*k/d)
        return b*(1 + (1 - (b*k/d)**(beta-1) )/(beta-1) )/(1 - b*k/d)
    
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, int(i*r), a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    return EC_given_kD_leq_d
  
  def approx_(d):
    # print("B(1-beta, 1, d/k)= {}".format(B(1-beta, 1, d/10) ) )
    E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
    # def E_D_given_D_leq_doverk(k):
    #   if b >= d/k:
    #     return 0
    #   return (b + b**beta*(B(1-beta, 1, d/k) - B(1-beta, 1, b) ) )/(1 - b*k/d)
    
    # EC_given_kD_leq_d = sum([EC_k_n_pareto(i, int(i*r), a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    Ek = k.mean()
    EC_given_kD_leq_d = EC_k_n_pareto(Ek, Ek*r, a, alpha)*E_D_given_D_leq_doverk(Ek)
    return EC_given_kD_leq_d
  
  d = 20
  blog(exact=exact(d), approx=approx(d), approx_=approx_(d) )

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
  # for x in np.linspace(l, u, 10):
  #   print("eq({})= {}".format(x, eq(x) ) )
  
  max_eq, u_w_max_eq = float('-inf'), 0
  u_w_max_eq
  eq_u = -1
  while u > l and eq_u < -0.01:
    eq_u = eq(u)
    if eq_u > max_eq:
      max_eq = eq_u
      u_w_max_eq = u
    u -= 0.05
  if u < l:
    print("u < l; u_w_max_eq= {}, max_eq= {}".format(u_w_max_eq, max_eq) )
    found_it = False
    for u in np.linspace(u_w_max_eq-0.05, u_w_max_eq+0.05, 10):
      if eq(u) > -0.01:
        found_it = True
        break
    if not found_it:
      return None
  print("l= {}, u= {}".format(l, u) )
  
  # eq_l, eq_u = eq(l), eq(u)
  # print("eq({})= {}, eq({})= {}".format(l, eq_l, u, eq_u) )
  # if eq_l*eq_u > 0:
  #   return None
  
  ro = scipy.optimize.brentq(eq, l, u)
  # ro = scipy.optimize.newton(eq, 1)
  # ro = scipy.optimize.fixed_point(ro_, 0.5)
  # ro = scipy.optimize.fixed_point(ro_, [0.01, 0.99] )
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
    a = 1.1/ro
    if a < 0:
      log(ERROR, "a= {} < 0!".format(a), ro=ro)
      a = float("inf")
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
  # for d in [11, 20]:
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

def test():
  N, Cap = 10, 100
  D = TPareto(1, 1000, 1) # Pareto(10, 2)
  k = BZipf(1, 10)
  
  def do_for(d):
    E_D_given_D_leq_doverk = lambda k_: mean(D, given_X_leq_x=True, x=d/k_)
    E_D_given_D_g_doverk = lambda k_: mean(D, given_X_leq_x=False, x=d/k_)
    
    E_kD_given_kD_leq_d = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    Pr_kD_leq_d = sum([D.cdf(d/i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in range(k.l_l, k.u_l+1) ] )
    
    E_kD = k.mean()*D.mean()
    E_kD_totalsum = Pr_kD_leq_d*E_kD_given_kD_leq_d + (1 - Pr_kD_leq_d)*E_kD_given_kD_g_d
    log(INFO, "d= {}".format(d), E_kD=E_kD, E_kD_totalsum=E_kD_totalsum)
  
  for d in np.linspace(10, 1000, 10):
    do_for(d)
  
if __name__ == "__main__":
  # plot_slowdown()
  # test()
  
  compare_exact_approx()
