import numpy as np
import mpmath, scipy
from rvs import *

# ##########################################  Basics  ############################################ #
def binom_(n, k):
  return scipy.special.binom(n, k)

def G(z, x=None, type_=None):
  if x is None:
    return scipy.special.gamma(z)
  else:
    if type_ == 'lower':
      return scipy.special.gammainc(z, x)*G(z)
    elif type_ == 'upper':
      # return (1 - scipy.special.gammainc(z, x) )*G(z)
      return scipy.special.gammaincc(z, x)*G(z)

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

def E_X_i_j_pareto(n, i, j, loc, a):
  if i > j:
    _j = j
    j = i
    i = _j
  if a <= max(2/(n-i+1), 1/(n-j+1) ):
    return 0 # None
  return loc**2*G(n+1)/G(n+1-2/a) * G(n-i+1-2/a)/G(n-i+1-1/a) * G(n-j+1-1/a)/G(n-j+1)

# ###############  Latency and Cost of zero-delay replicated or coded redundancy  ################ #
def ES_k_c_pareto(k, c, loc, a):
  return loc*G(k+1)*G(1-1/(c+1)/a)/G(k+1-1/(c+1)/a)

def EC_k_c_pareto(k, c, loc, a):
  return k*(c+1) * a*(c+1)*loc/(a*(c+1)-1)

def ES2_k_c_pareto(k, c, loc, a):
  a_ = (c+1)*a
  if a_ > 1:
    return E_X_i_j_pareto(k, k, k, loc, a_)
  else:
    return None

def EC2_k_c_pareto(k, c, loc, a):
  a_ = (c+1)*a
  # if a_ > 2:
  #   return (k*(c+1))**2 * loc**2*a_/(a_-2)
  # else:
  #   None
  EC2 = 0
  for i in range(1, k+1):
    for j in range(1, k+1):
      EC2 += E_X_i_j_pareto(k, i, j, loc, a_)

  return (c+1)**2 * EC2

def ES_k_n_pareto(k, n, loc, a):
  # log(INFO, "", n=n, k=k, loc=loc, a=a)
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

def ES2_k_n_pareto(k, n, loc, a):
  return E_X_i_j_pareto(n, k, k, loc, a)

def EC2_k_n_pareto(k, n, loc, a):
  EC2 = (n-k)**2*E_X_i_j_pareto(n, k, k, loc, a)
  for i in range(1, k+1):
    EC2 += 2*(n-k)*E_X_i_j_pareto(n, i, k, loc, a)
  for i in range(1, k+1):
    for j in range(1, k+1):
      EC2 += E_X_i_j_pareto(n, i, j, loc, a)
  
  return EC2

# ###########################################  Qing  ############################################# #
def MGc_EW_Prqing(ar, c, EX, EX2):
  def MMc_EW_Prqing(ar, EX, c):
    ro = ar*EX/c
    log(INFO, "c= {}, ro= {}".format(c, ro) )
    ro_ = c*ro
    # Prqing = 1/(1 + (1-ro)*G(c+1)/ro_**c * sum([ro_**i/G(i+1) for i in range(c) ] ) )
    c_times_ro__power_c = math.exp(c*math.log(c*ro) )
    # Prqing = 1/(1 + (1-ro) * math.exp(ro_)*G(c, ro_, 'upper')/c_times_ro__power_c)
    Prqing = 1/(1 + (1-ro) * c*math.exp(ro_)*G(c, ro_, 'upper')/c_times_ro__power_c)
    
    # EN = ro/(1-ro)*Prqing + c*ro
    # log(INFO, "ro= {}, Prqing= {}".format(ro, Prqing) )
    return Prqing/(c/EX - ar), Prqing
  # CoeffVar = math.sqrt(EX2 - EX**2)/EX
  # return (1 + CoeffVar**2)/2 * MMc_EW_Prqing(ar, EX, c)
  MMc_EW, MMc_Prqing = MMc_EW_Prqing(ar, EX, c)
  return (1 + (EX2 - EX**2)/EX**2)/2 * MMc_EW, MMc_Prqing
 