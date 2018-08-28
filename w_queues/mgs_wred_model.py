from rvs import *

def mmn_prob_jinsys(n, ar, mu, j):
  ro = ar/mu
  P0 = 1/(ro**n/fact(n)*(n*mu/(n*mu - ar) ) + sum([ro**i/fact(i) for i in range(n) ] ) )
  if j <= n:
    return ro**j/fact(j) * P0
  else:
    return ro**j/fact(n)/(n)**(j-n) * P0

def mmn_ENq(n, ar, mu):
  ro = ar/mu
  P0 = 1/(ro**n/fact(n)*(n*mu/(n*mu - ar) ) + sum([ro**i/fact(i) for i in range(n) ] ) )
  print("P0= {}".format(P0) )
  
  ENq = P0*ro**(n+1)/fact(n-1)/(n - ro)**2 # P0*ar*mu*ro**(n+1)/fact(n-1)/(n*mu - ar)**2
  return ENq

def mmn_ET(n, ar, mu):
  ro = ar/mu
  EN = mmn_ENq(n, ar, mu) + ro
  return EN/ar

def mgn_ET(n, ar, V):
  EV = moment_ith(V, 1)
  ro = ar*EV
  cvar_V = moment_ith(V, 2)/EV**2 - 1
  ENq = (1 + cvar_V)/2 * mmn_ENq(n, ar, 1/EV)
  EN = ENq + ro
  return EN/ar

def mgn_rep_ET(n, ar, V):
  EV = moment_ith(V, 1)
  # return 1/(n/EV - ar)
  # p_l = [ar**(-i) for i in range(n) ]
  
  ar_ub = n/EV
  # p_l = [(ar*1/(ar_ub - ar) )**(n-i) for i in range(1, n+1) ]
  f, l = 1/(ar_ub - ar), 1/ar
  a = (l/f)**(1/(n-1))
  p_l = [f*a**i for i in range(n) ]
  s = sum(p_l)
  p_l = [p/s for p in p_l]
  print("p_l= {}".format(p_l) )
  
  EG, EG2 = 0, 0
  for i in range(1, n+1):
    Gi = X_n_k(V, i, 1)
    EG += p_l[i-1]*moment_ith(Gi, 1)
    EG2 += p_l[i-1]*moment_ith(Gi, 2)
  
  ro = ar*EG
  cvar_G = EG2/EG**2 - 1
  ENq = (1 + cvar_G)/2 * mmn_ENq(n, ar, 1/EG)
  EN = ENq + ro
  return EN/ar
