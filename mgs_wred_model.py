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
  EN = mmn_ENq(n, ar, mu) + ro
  return EN/ar

def mgn_ET(n, ar, V):
  EV = moment_ith(V, 1)
  ro = ar*EV
  cvar_V = moment_ith(V, 2)/EV**2 - 1
  ENq = (1 + cvar_V)/2 * mmn_ENq(n, ar, 1/EV)
  EN = ENq + ro
  return EN/ar
