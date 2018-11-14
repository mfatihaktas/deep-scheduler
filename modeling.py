import numpy as np
import mpmath, scipy

from rvs import *
from plot_utils import *

# ##########################################  Basics  ############################################ #
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

def ET_k_c_pareto(k, c, loc, a):
  return loc*G(k+1)*G(1-1/(c+1)/a)/G(k+1-1/(c+1)/a)

def EC_k_c_pareto(k, c, loc, a):
  return k*(c+1) * a*(c+1)*loc/(a*(c+1)-1)

def ET_k_n_pareto(k, n, loc, a):
  if k == 0:
    return 0
  elif n == k and n > 170:
    return loc*(k+1)**(1/a) * G(1-1/a)
  elif n > 170:
    return loc*((n+1)/(n-k+1))**(1/a)
  return loc*G(n+1)/G(n-k+1)*G(n-k+1-1/a)/G(n+1-1/a)

def EC_k_n_pareto(k, n, loc, a):
  if n > 170:
    return loc/(a-1) * (a*n - (n-k)*((n+1)/(n-k+1))**(1/a) )
  return loc*n/(a-1) * (a - G(n)/G(n-k)*G(n-k+1-1/a)/G(n+1-1/a) )

# #########################################  Modeling  ########################################### #
'''
Kubernetes architecture; master dispatches arriving jobs to distributed workers.
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
def Pr_kD_leq_d_pareto(k, b, beta, d):
  # D = Pareto(b, beta)
  # return sum([D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
  def Pr_D_leq_doverk(k):
    if b <= d/k:
      return 1 - (b*k/d)**beta
    else:
      return 0
  return sum([Pr_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )

def EC_exact_pareto(k, r, b, beta, a, alpha, d, red):
  D = Pareto(b, beta)
  S = Pareto(a, alpha)
  if d is None:
    return k.mean()*S.mean()*D.mean()
  
  ES = S.mean()
  
  E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, int(i*r), a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, int(r)-1, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  EC_given_kD_g_d = ES*sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

# D ~ Pareto(b, beta), S ~ Pareto(a, alpha)
def EC_approx_pareto(k, r, b, beta, a, alpha, d=None, red=None):
  ES = a/(1 - 1/alpha)
  if d is None:
    ED = b/(1 - 1/beta)
    return k.mean()*ES*ED
  
  def E_D_given_D_leq_doverk(k):
    if b >= d/k:
      # return d/k/(1 - (b*k/d)**beta)
      return 0
    else:
      # return (b + b**beta*(B(1-beta, 1, d/k) - B(1-beta, 1, b) ) )/(1 - (b*k/d)**beta)
      return b*(1 + (1 - (b*k/d)**(beta-1) )/(beta-1) )/(1 - (b*k/d)**beta)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, int(i*r), a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, int(r)-1, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  def E_D_given_D_g_doverk(k):
    # result, abserr = scipy.integrate.quad(D.tail, 0, d/k)
    # return (D.mean() - result)/D.tail(d/k)
    if b > d/k:
      return b/(1 - 1/beta) - d/k
    else:
      # result, abserr = scipy.integrate.quad(D.tail, 0, d/k)
      # return (D.mean() - result)/D.tail(d/k)
      # return (b*beta/(beta-1) - result)/(b*k/d)**beta
      return d/(beta-1)/k
  EC_given_kD_g_d = ES*sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  # EC_given_kD_g_d = ES*d/(beta-1)
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

def compare_EC_exact_approx():
  # N, Cap = 10, 100
  red = 'Coding'
  k = BZipf(1, 10)
  r = 1.5
  b, beta = 10, 1.1
  a, alpha = 1, 20
  log(INFO, "", red=red, k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  
  for d in [None, *np.linspace(0.1, 10, 10), *np.linspace(100, 1000, 10) ]:
    print(">> d= {}".format(d) )
    blog(EC_exact=EC_exact_pareto(k, r, b, beta, a, alpha, d, red),
         EC_approx=EC_approx_pareto(k, r, b, beta, a, alpha, d, red) )


def ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  def func_ro(ro):
    # return ar/N/Cap * EC_exact_pareto(k, r, b, beta, a, alpha_gen(ro), d)
    return ar/N/Cap * EC_approx_pareto(k, r, b, beta, a, alpha_gen(ro), d, red)
  
  eq = lambda ro: ro - func_ro(ro)
  l, u = 0.0001, 1
  # max_eq, u_w_max_eq = float('-inf'), 0
  # u_w_max_eq
  # eq_u = -1
  # while u > l and eq_u < -0.01:
  #   eq_u = eq(u)
  #   if eq_u > max_eq:
  #     max_eq = eq_u
  #     u_w_max_eq = u
  #   u -= 0.05
  # if u < l:
  #   print("u < l; u_w_max_eq= {}, max_eq= {}".format(u_w_max_eq, max_eq) )
  #   found_it = False
  #   for u in np.linspace(u_w_max_eq-0.05, u_w_max_eq+0.05, 10):
  #     if eq(u) > -0.01:
  #       found_it = True
  #       break
  #   if not found_it:
  #     return None
  print("l= {}, u= {}".format(l, u) )
  try:
    ro = scipy.optimize.brentq(eq, l, u)
  except ValueError:
    return None
  # ro = scipy.optimize.newton(eq, 1)
  # ro = scipy.optimize.fixed_point(ro_, 0.5)
  # ro = scipy.optimize.fixed_point(ro_, [0.01, 0.99] )
  return ro

def ar_for_ro_pareto(ro, N, Cap, k, b, beta, a, alpha_gen):
  D = Pareto(b, beta)
  S = Pareto(a, alpha_gen(ro) )
  return ro*N*Cap/k.mean()/D.mean()/S.mean()

def Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  alpha = alpha_gen(ro)
  if d is None:
    return sum([ET_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  # log(INFO, "", d=d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  # S = Pareto(a, alpha_gen(ro) )
  # E_S_given_kD_g_d = sum([X_n_k(S, i, i).mean()*k.pdf(i) for i in k.v_l] )
  # E_S_given_kD_leq_d = sum([X_n_k(S, int(i*r), i).mean()*k.pdf(i) for i in k.v_l] )
  # return E_S_given_kD_leq_d*Pr_kD_leq_d + \
  #       E_S_given_kD_g_d*(1 - Pr_kD_leq_d)
  
  E_S_given_kD_g_d = sum([ET_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  E_S_given_kD_leq_d = sum([ET_k_n_pareto(i, int(i*r), a, alpha)*k.pdf(i) for i in k.v_l] )
  return E_S_given_kD_leq_d*Pr_kD_leq_d + \
         E_S_given_kD_g_d*(1 - Pr_kD_leq_d)

def plot_ro_Esl():
  N, Cap = 10, 100
  b, beta = 10, 1.1
  a, alpha = 1, 2 # 2.1
  # k = BZipf(1, 10)
  # r = 1.5
  k = BZipf(1, 1) # DUniform(1, 1)
  r = 2 # 1.5
  log(INFO, "", k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  
  def alpha_gen(ro):
    # return alpha
    return alpha/ro
    # return alpha - ro
  
  ar = ar_for_ro_pareto(1/2, N, Cap, k, b, beta, a, alpha_gen)
  print("ar= {}".format(ar) )
  
  d = None
  ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d)
  Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d)
  print("\n>> d= {}".format(d) )
  blog(ro=ro, Esl=Esl)
  
  d_l = []
  ro_wrep_l, Esl_wrep_l = [], []
  ro_wcoding_l, Esl_wcoding_l = [], []
  l, u = a*b, 1000
  for d in np.logspace(math.log10(l), math.log10(u), 40):
    print("\n>> d= {}".format(d) )
    d_l.append(d)
    
    red = 'Rep'
    ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
    blog(ro=ro, Esl=Esl)
    ro_wrep_l.append(ro)
    Esl_wrep_l.append(Esl)
    
    red = 'Coding'
    ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
    blog(ro=ro, Esl=Esl)
    ro_wcoding_l.append(ro)
    Esl_wcoding_l.append(Esl)
  # 
  fig, axs = plot.subplots(1, 2)
  fontsize = 14
  ax = axs[0]
  plot.sca(ax)
  plot.plot(d_l, ro_wrep_l, label='w/ Rep', c='blue', marker=next(marker_c), ls=':', mew=1)
  plot.plot(d_l, ro_wcoding_l, label='w/ Coding', c=NICE_BLUE, marker=next(marker_c), ls=':', mew=1)
  prettify(ax)
  plot.xscale('log')
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('Average load', fontsize=fontsize)
  ax = axs[1]
  plot.sca(ax)
  plot.plot(d_l, Esl_wrep_l, label='w/ Rep', c='blue', marker=next(marker_c), ls=':', mew=1)
  plot.plot(d_l, Esl_wcoding_l, label='w/ Coding', c=NICE_RED, marker=next(marker_c), ls=':', mew=1)
  prettify(ax)
  plot.xscale('log')
  # plot.legend()
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('Average slowdown', fontsize=fontsize)
  
  plot.subplots_adjust(hspace=2)
  st = plot.suptitle(r'$N= {}$, $C= {}$, $k \sim$ {}, r= {}'.format(N, Cap, k, r) + '\n' + r'$b= {}$, $\beta= {}$, $a= {}$, $\alpha= {}$'.format(b, beta, a, alpha) )
  plot.savefig('plot_ro_Esl.png', bbox_extra_artists=(st,), bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

if __name__ == "__main__":
  # plot_slowdown()
  # test()
  
  # compare_EC_exact_approx()
  plot_ro_Esl()
