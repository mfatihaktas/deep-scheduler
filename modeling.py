import numpy as np
import mpmath, scipy, collections

from rvs import *
from plot_utils import *
from sim_objs_lessreal import *

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
  D = Pareto(b, beta)
  return sum([D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
  
  # def Pr_D_leq_doverk(k):
  #   if b <= d/k:
  #     return 1 - (b*k/d)**beta
  #   else:
  #     return 0
  # return sum([Pr_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )

def test():
  N, Cap = 4, 2
  b, beta = 10, 1.1
  a, alpha = 1, 2
  k = BZipf(1, 2)
  r = 2
  log(INFO, "", k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  
  D = Exp(beta, b) # Pareto(b, beta)
  S = Exp(alpha, a) # Pareto(a, alpha)
  Ek = k.mean()
  ES = S.mean()
  ED = D.mean()
  
  def gen_sim_E_kD_given_kD_leq_d(d, nsamples=100*1000):
    sum_sample = 0
    for _ in range(nsamples):
      k_sample = k.sample()
      D_sample = D.sample()
      kD_sample = k_sample*D_sample
      if kD_sample <= d:
        sum_sample += kD_sample
    return sum_sample/nsamples
  
  def Pr_kD(x):
    return sum([D.pdf(x/i)*k.pdf(i) for i in k.v_l] )
  def Pr_kD_leq_x(x):
    return sum([D.cdf(x/i)*k.pdf(i) for i in k.v_l] )
  
  E_kD = Ek*ED
  # E_kD_ = mpmath.quad(lambda x: x*Pr_kD(x), [0, mpmath.inf] )
  # E_kD__ = mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, mpmath.inf] )
  E_kD_ = scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, np.inf)[0]
  E_kD__ = scipy.integrate.quad(lambda x: 1 - Pr_kD_leq_x(x), 0, np.inf)[0]
  print("E_kD= {}, E_kD_= {}, E_kD__= {}".format(E_kD, E_kD_, E_kD__) )
  # 
  def compute(d):
    Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
    
    # mpmath.quad(lambda x: x*Pr_kD(x), [0, d] ) \
    E_kD_given_kD_leq_d = scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] \
                        / Pr_kD_leq_d if Pr_kD_leq_d != 0 else 0
    
    # E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
    # E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
    E_kD_given_kD_g_d = (Ek*ED - scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] ) \
                      / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
    
    log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + E_kD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
    blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, E_kD_given_kD_g_d=E_kD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
    
    # Using law of total expectation
    E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
    E_kD_given_kD_leq_d_ = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
    E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
    E_kD_given_kD_g_d_ = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
    blog(E_kD_given_kD_leq_d_=E_kD_given_kD_leq_d_, E_kD_given_kD_g_d_=E_kD_given_kD_g_d_)
  
    sim_E_kD_given_kD_leq_d = gen_sim_E_kD_given_kD_leq_d(d)
    blog(sim_E_kD_given_kD_leq_d=sim_E_kD_given_kD_leq_d)
  
  l, u = a*b, 1000
  for d in np.logspace(math.log10(l), math.log10(u), 10):
    print("\n>> d= {}".format(d) )
    compute(d)
  
  log(INFO, "done.")

def EC_exact_pareto(k, r, b, beta, a, alpha, d, red):
  D = Pareto(b, beta)
  S = Pareto(a, alpha)
  if d is None:
    return k.mean()*S.mean()*D.mean()
  
  Ek = k.mean()
  ES = S.mean()
  ED = D.mean() # b/(1 - 1/beta)
  
  E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, i*r, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, r - 1, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  E_kD_given_kD_leq_d = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # def Pr_kD(x):
  #   return sum([D.pdf(x/i)*k.pdf(i) for i in k.v_l] )
  # mpmath.quad(lambda x: x*Pr_kD(x), [0, d] ) \
  # E_kD_given_kD_leq_d = scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] \
  #                     / Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  
  E_D_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  # E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  E_kD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  
  # E_kD_given_kD_g_d = (Ek*ED - scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] ) \
  #                   / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  # def Pr_kD_leq_x(x):
  #   return sum([D.cdf(x/i)*k.pdf(i) for i in k.v_l] )
  # E_kD_given_kD_g_d = (Ek*ED - mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] ) )/(1 - Pr_kD_leq_d)
  
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  
  # E_kD_given_kD_leq_d = mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] )/Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  # E_kD_given_kD_g_d = (Ek*ED - mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] ) )/(1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  
  # E_kD_given_kD_leq_d = sum([mpmath.quad(lambda x: D.tail(x), [0, d/i] )*i*k.pdf(i) for i in k.v_l] ) \
  #                     / Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  # E_kD_given_kD_g_d = (Ek*ED - sum([mpmath.quad(lambda x: D.tail(x), [0, d/i] )*i*k.pdf(i) for i in k.v_l] ) ) \
  #                   / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  
  # E_kD_given_kD_g_d = (Ek*ED - E_kD_given_kD_leq_d*Pr_kD_leq_d)/(1 - Pr_kD_leq_d)
  EC_given_kD_g_d = ES*E_kD_given_kD_g_d
  
  # log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + E_kD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
  # blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, E_kD_given_kD_g_d=E_kD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

# D ~ Pareto(b, beta), S ~ Pareto(a, alpha)
def EC_model_pareto(k, r, b, beta, a, alpha, d=None, red=None):
  Ek = k.mean()
  ED = b/(1 - 1/beta)
  ES = a/(1 - 1/alpha)
  if d is None:
    return Ek*ES*ED
  
  def E_D_given_D_leq_doverk(k):
    if b >= d/k:
      return 0
    else:
      return beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) / (1 - (b*k/d)**beta)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, i*r, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, r - 1, a, alpha)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  def E_D_given_D_g_doverk(k):
    ED = b/(1 - 1/beta)
    if b > d/k:
      return ED
    else:
      return (ED - beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) ) / (b*k/d)**beta
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  E_kD_given_kD_leq_d = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  E_kD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  EC_given_kD_g_d = ES*E_kD_given_kD_g_d
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

def EC_approx_pareto(k, r, b, beta, a, alpha, d=None, red=None):
  D = Pareto(b, beta)
  ED = b/(1 - 1/beta)
  ES = a/(1 - 1/alpha)
  Ek = k.mean()
  if d is None:
    return Ek*ES*ED
  
  E_D_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  E_kD_given_kD_leq_d = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  
  def E_D_given_D_g_doverk(k):
    ED = b/(1 - 1/beta)
    if b > d/k:
      return ED
    else:
      return (ED - beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) ) / (b*k/d)**beta
  # E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  E_kD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  
  def f():
    if red == 'Coding':
      return a*r/(alpha-1) * (alpha - (1 - 1/r)**(1 - 1/alpha) )
    elif red == 'Rep':
      return a*r*alpha*r/(alpha*r - 1)
  E_kD = Ek*ED
  E_kD_ = E_kD_given_kD_leq_d*Pr_kD_leq_d + E_kD_given_kD_g_d*(1 - Pr_kD_leq_d)
  log(INFO, "E_kD= {}, E_kD_= {}".format(E_kD, E_kD_) )
  
  # return f()*E_kD_given_kD_leq_d*Pr_kD_leq_d + ES*E_kD_given_kD_g_d*(1 - Pr_kD_leq_d)
  return Ek*ES*ED + E_kD_given_kD_leq_d*Pr_kD_leq_d*(f() - ES)
  
  '''
  # Debugging
  # EC_given_kD_leq_d = sum([i*f()*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # EC_given_kD_leq_d = sum([EC_k_c_pareto(i, int(r)-1, a, alpha) * E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # def EC_k_c_pareto(k, c, loc, a):
  #   return k*(c+1) * a*(c+1)*loc/(a*(c+1)-1)
  # EC_given_kD_leq_d = sum([i*r * alpha*r*a/(alpha*r-1) * E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  
  EC_given_kD_leq_d = f()*E_kD_given_kD_leq_d
  
  def E_D_given_D_g_doverk(k):
    if b > d/k:
      # return b/(1 - 1/beta) - d/k
      return ED
    else:
      return d/(beta-1)/k
  E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  EC_given_kD_g_d = ES*E_kD_given_kD_g_d
  
  ED = b/(1 - 1/beta)
  log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + E_kD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
  blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, E_kD_given_kD_g_d=E_kD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)
  '''

def compare_EC_exact_approx():
  # N, Cap = 10, 100
  red = 'Rep' # 'Coding'
  k = BZipf(1, 10)
  r = 2
  b, beta = 10, 1.1
  a, alpha = 1, 2
  log(INFO, "", red=red, k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  
  for d in [None, *np.linspace(0.1, 10, 4), *np.linspace(100, 1000, 20) ]:
    print(">> d= {}".format(d) )
    blog(EC_exact=EC_exact_pareto(k, r, b, beta, a, alpha, d, red),
         EC_model=EC_model_pareto(k, r, b, beta, a, alpha, d, red),
         EC_approx=EC_approx_pareto(k, r, b, beta, a, alpha, d, red) )

def ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  def func_ro(ro):
    # return ar/N/Cap * EC_exact_pareto(k, r, b, beta, a, alpha_gen(ro), d)
    return ar/N/Cap * EC_model_pareto(k, r, b, beta, a, alpha_gen(ro), d, red)
    # return ar/N/Cap * EC_approx_pareto(k, r, b, beta, a, alpha_gen(ro), d, red)
  
  eq = lambda ro: ro - func_ro(ro)
  l, u = 0.0001, 1
  
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
  if red == 'Coding':
    E_S_given_kD_leq_d = sum([ET_k_n_pareto(i, i*r, a, alpha)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    E_S_given_kD_leq_d = sum([ET_k_c_pareto(i, r-1, a, alpha)*k.pdf(i) for i in k.v_l] )
  return E_S_given_kD_leq_d*Pr_kD_leq_d + \
         E_S_given_kD_g_d*(1 - Pr_kD_leq_d)

def plot_ro_Esl():
  N, Cap = 10, 100
  b, beta = 1, 1.1
  a, alpha = 1, 1.45
  k = BZipf(1, 20)
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
  print("ro= {}".format(ro) )
  Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d)
  print("\n>> d= {}".format(d) )
  blog(ro=ro, Esl=Esl)
  
  d_l = []
  ro_wrep_l, Esl_wrep_l = [], []
  ro_wcoding_l, Esl_wcoding_l = [], []
  l, u = a*b, 10**4
  for d in np.logspace(math.log10(l), math.log10(u), 20):
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
  plot.legend()
  plot.xscale('log')
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('Avg load', fontsize=fontsize)
  ax = axs[1]
  plot.sca(ax)
  plot.plot(d_l, Esl_wrep_l, label='w/ Rep', c='blue', marker=next(marker_c), ls=':', mew=1)
  plot.plot(d_l, Esl_wcoding_l, label='w/ Coding', c=NICE_RED, marker=next(marker_c), ls=':', mew=1)
  prettify(ax)
  plot.xscale('log')
  plot.legend()
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('Avg slowdown at service time', fontsize=fontsize)
  
  plot.subplots_adjust(hspace=2)
  st = plot.suptitle(r'$N= {}$, $C= {}$, $k \sim$ {}, r= {}'.format(N, Cap, k, r) + '\n' + r'$b= {}$, $\beta= {}$, $a= {}$, $\alpha= {}$'.format(b, beta, a, alpha) )
  plot.gcf().set_size_inches(2*5, 5)
  plot.savefig('plot_ro_Esl.png', bbox_extra_artists=(st,), bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

# ###########################################  Sim  ############################################## #
def sim(sinfo_m, mapping_m, sching_m, plotname_suffix=''):
  env = simpy.Environment()
  cl = Cluster_LessReal(env, scher=Scher_wMultiplicativeExpansion(mapping_m, sching_m), **sinfo_m)
  jg = JobGen_LessReal(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  fig, axs = plot.subplots(len(cl.w_l), 1, sharex='col')
  if len(cl.w_l) == 1:
    axs = [axs]
  avg_schedload_l = []
  for i, w in enumerate(cl.w_l):
    print("w._id= {}, w.avg_load= {}".format(w._id, w.avg_load() ) )
    avg_schedload_l.append(w.avg_load() )
    
    plot.sca(axs[i] )
    t_l, t_load_l = map_to_key__val_l(w.t_load_m)
    plot.plot(t_l, t_load_l, label='w.id= {}'.format(w._id), color=next(darkcolor_c), marker=next(marker_c), linestyle='None')
    plot.ylabel('Load')
    plot.legend()
    plot.xticks(rotation=70)
    plot.xlabel('Time (sec)')
  fig.set_size_inches(2*8, len(cl.w_l)*4)
  plot.subplots_adjust(hspace=0.25, wspace=0.25)
  plot.savefig('plot_wloadovertime_{}_{}.png'.format(sching_m['type'], plotname_suffix), bbox_inches='tight')
  
  njobs_wfate, ndropped = 0, 0
  njobs_waited_inq = 0
  waittime_l = []
  sl_l, serv_sl_l = [], []
  for jid, info in cl.jid_info_m.items():
    if 'fate' in info:
      njobs_wfate += 1
      fate = info['fate']
      if fate == 'dropped':
        ndropped += 1
      elif fate == 'finished':
        serv_sl_l.append(info['run_time']/info['expected_run_time'] )
        sl_l.append(
          (info['wait_time'] + info['run_time'] )/info['expected_run_time'] )
        waittime_l.append(info['wait_time'] )
        if info['wait_time'] > 0:
          njobs_waited_inq += 1
  frac_jobs_waited_inq = njobs_waited_inq/len(cl.jid_info_m)
  blog(ndropped=ndropped, njobs_wfate=njobs_wfate, frac_jobs_waited_inq=frac_jobs_waited_inq)
  
  return {
    'drop_rate': ndropped/len(cl.jid_info_m),
    'waittime_mean': np.mean(waittime_l),
    'sl_mean': np.mean(sl_l),
    'sl_std': np.std(sl_l),
    'serv_sl_mean': np.mean(serv_sl_l),
    'load_mean': np.mean(avg_schedload_l) }

def plot_sim():
  blog(sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  def plot_wrt_d():
    d_l = []
    # ro_wrep_l, Esl_wrep_l = [], []
    ro_wcoding_l, Esl_wcoding_l = [], []
    sim_ro_wcoding_l, sim_Esl_wcoding_l = [], []
    l, u = a*b, 1000
    for d in np.logspace(math.log10(l), math.log10(u), 5):
      d = round(d, 2)
      print("\n>> d= {}".format(d) )
      d_l.append(d)
      
      # red = 'Rep'
      # ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      # Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
      # blog(ro=ro, Esl=Esl)
      # ro_wrep_l.append(ro)
      # Esl_wrep_l.append(Esl)
      
      red = 'Coding'
      ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      Esl = Esl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
      blog(ro=ro, Esl=Esl)
      ro_wcoding_l.append(ro)
      Esl_wcoding_l.append(Esl)
      
      sching_m['threshold'] = d
      sim_m = sim(sinfo_m, mapping_m, sching_m, 'd{}'.format(sching_m['threshold'] ) )
      blog(sim_m=sim_m)
      sim_ro = sim_m['load_mean']
      sim_Esl = sim_m['serv_sl_mean']
      # blog(sim_ro=sim_ro, sim_Esl=sim_Esl)
      sim_ro_wcoding_l.append(sim_ro)
      sim_Esl_wcoding_l.append(sim_Esl)
    # 
    fig, axs = plot.subplots(1, 2)
    fontsize = 14
    ax = axs[0]
    plot.sca(ax)
    # plot.plot(d_l, ro_wrep_l, label='w/ Rep', c='blue', marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, ro_wcoding_l, label='w/ Coding', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, sim_ro_wcoding_l, label='Sim, w/ Coding', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    prettify(ax)
    plot.legend()
    plot.xscale('log')
    plot.xlabel('d', fontsize=fontsize)
    plot.ylabel('Average load', fontsize=fontsize)
    ax = axs[1]
    plot.sca(ax)
    # plot.plot(d_l, Esl_wrep_l, label='w/ Rep', c='blue', marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, Esl_wcoding_l, label='w/ Coding', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    plot.plot(d_l, sim_Esl_wcoding_l, label='Sim, w/ Coding', c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
    prettify(ax)
    plot.xscale('log')
    plot.legend()
    plot.xlabel('d', fontsize=fontsize)
    plot.ylabel('Average slowdown', fontsize=fontsize)
  
  def plot_wrt_ar():
    sching_m['threshold'] = float('Inf')
    ar_ub = ar_for_ro_pareto(0.9, N, Cap, k, b, beta, a, alpha_gen)
    sim_ro_wcoding_l, sim_Esl_wcoding_l = [], []
    for ar in np.linspace(ar_ub/10, ar_ub, 5):
      sinfo_m['ar'] = ar
      sim_m = sim(sinfo_m, mapping_m, sching_m)
      blog(sim_m=sim_m)
      sim_ro = sim_m['load_mean']
      sim_Esl = sim_m['serv_sl_mean']
      # blog(sim_ro=sim_ro, sim_Esl=sim_Esl)
      sim_ro_wcoding_l.append(sim_ro)
      sim_Esl_wcoding_l.append(sim_Esl)
  
  plot.subplots_adjust(hspace=2)
  st = plot.suptitle(r'$N= {}$, $C= {}$, $k \sim$ {}, r= {}'.format(N, Cap, k, r) + '\n' + r'$b= {}$, $\beta= {}$, $a= {}$, $\alpha= {}$'.format(b, beta, a, alpha) )
  plot.gcf().set_size_inches(2*5, 5)
  plot.savefig('plot_ro_Esl.png', bbox_extra_artists=(st,), bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

if __name__ == "__main__":
  N, Cap = 10, 1
  b, beta = 10, 5
  a, alpha = 1, 1000 # 2
  k = BZipf(1, 1)
  r = 1
  log(INFO, "", k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  def alpha_gen(ro):
    return alpha
  S = Pareto(a, alpha)
  ar = round(ar_for_ro_pareto(1/2, N, Cap, k, b, beta, a, alpha_gen), 2)
  
  sinfo_m = {
    'ar': ar, 'njob': 2000*5, 'nworker': N, 'wcap': Cap,
    'totaldemand_rv': Pareto(b, beta),
    'demandperslot_mean_rv': DUniform(1, 1),
    'k_rv': k,
    'straggle_m': {'slowdown': lambda load: S.sample() } }
  mapping_m = {'type': 'spreading'}
  sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  blog(sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  # plot_slowdown()
  
  # test()
  # compare_EC_exact_approx()
  # plot_ro_Esl()
  
  plot_sim()
