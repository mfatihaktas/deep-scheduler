import numpy as np

from patch import *
from rvs import *

"""
def sim_gain(V_rv, n, k, num_srun):
  g = 0
  for _ in range(num_srun):
    g += V_rv.gen_sample() - gen_orderstat_sample(V_rv, n, k)
  return g/num_srun

def sim_pain(V_rv, n, k, X_rv, num_srun):
  def sim():
    j, s, cost = 0, 0, 0
    t = gen_orderstat_sample(V_rv, n, k)
    while s < t:
      s += X_rv.gen_sample()
      j += 1
      if s < t:
        cost += t - s
    return cost, j
  
  cost_l, j_l = [], []
  for _ in range(num_srun):
    cost, j = sim()
    cost_l.append(cost)
    j_l.append(j)
  EC = np.mean(cost_l)
  EJ = np.mean(j_l)
  EJ2 = np.mean([j**2 for j in j_l] )
  # VJ = np.mean([(j - EJ)**2 for j in j_l] )
  
  return {'EC': EC,
          'EJ': EJ, 'EJ2': EJ2}

def Ecost_model(T_rv, X_rv):
  ET, ET2 = moment_ith(T_rv, 1), moment_ith(T_rv, 2)

def EJ_model(T_rv, X_rv): # X ~ Exp
  return 1 + moment_ith(T_rv, 1)/moment_ith(X_rv, 1)

def EJ2_model(T_rv, X_rv): # X ~ Exp
  ET, ET2 = moment_ith(T_rv, 1), moment_ith(T_rv, 2)
  # VT = moment_ith(T_rv, 2) - ET**2
  
  EX, EX2, EX3 = moment_ith(X_rv, 1), moment_ith(X_rv, 2), moment_ith(X_rv, 3)
  VX = EX2 - EX**2
  K = VX*EX2/2/EX**2 + 3/4*(EX2/EX)**2 - 2/3*EX3/EX
  return ET*VX/EX**3 + K/EX**2 + (1 + 2*ET/EX + ET2/EX**2)

def EC_model(X_rv, V_rv, m, c): # X ~ Exp
  X = X_n_k(X_rv, c-1, 1)
  EX = moment_ith(X, 1)
  
  T_rv = X_n_k(V_rv, c, 1)
  ET, ET2 = moment_ith(T_rv, 1), moment_ith(T_rv, 2)
  # EJ, EJ2 = EJ_model(T_rv, X), EJ2_model(T_rv, X)
  return m*ET + ET2/EX/2

def EC_model_VPareto(X_rv, V_rv, m, d):
  s, a = V_rv.loc, d*V_rv.a
  ar = X_rv.mu
  ET, ET2 = s*a/(a-1), s**2*a/(a-2)
  return m*ET + ar*ET2/2

def EP_model(X_rv, V_rv, m, c):
  return EC_model(X_rv, V_rv, m, c)
  
def EG_model(V_rv, c):
  return moment_ith(V_rv, 1) - moment_ith_n_k(V_rv, 1, c, 1)

def ar_max_for_EP_m_EG_g_zero(s, a, m, d): # V ~ Pareto(s, a), X ~ Exp
  return 2*(d*a - 2)/(d-1)/s/d * (1/(a - 1) - (m + 1)*d/(d*a - 1) )
  
def plot_gain_pain():
  num_srun = 10000 # *10
  s, a = 1, 2.1
  # V_rv = Exp(a) # Exp(a, D=s)
  V_rv = Pareto(s, a) # Dolly() # # TPareto(s, 12*s, 1) # Pareto(s, a)
  
  def plot_vs_a(c, m=0):
    EG = EG_model(V_rv, c)
    
    ar_l, EP_m_EG_l = [], []
    for ar in np.linspace(0.01, 1, 10):
      ar_l.append(ar)
      X_rv = Exp(ar)
      
      # EC = sim_pain(V_rv, c, 1, X_rv, num_srun)['EC']
      EP = EP_model(X_rv, V_rv, m, c)
      EP_m_EG_l.append(EP - EG)
    color = next(dark_color)
    # ar_max = ar_max_for_EP_m_EG_g_zero(s, a, m, c)
    # EP = EP_model(Exp(ar_max), V_rv, m, c)
    # plot.plot([ar_max], [EP-EG], label=r'$\lambda_{}$, c= {}'.format('\max', c), color=color, marker='x', mew=4, ms=10)
    plot.plot(ar_l, EP_m_EG_l, label=r'$E[P-G]$, c= {}'.format(c), color=color, marker=next(marker), linestyle=':', mew=2)
    plot.xlabel(r'$\lambda$', fontsize=14)
  def plot_vs_c(ar, m=0):
    X_rv = Exp(ar)
    c_l, EP_m_EG_l = [], []
    for c in np.arange(2, 8):
      c_l.append(c)
      EP = EP_model(X_rv, V_rv, m, c)
      EG = EG_model(V_rv, c)
      EP_m_EG_l.append(EP - EG)
    plot.plot(c_l, EP_m_EG_l, label=r'$E[P-G]$, ar= {}'.format(ar), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    plot.xlabel(r'$c$', fontsize=14)
  
  # plot_vs_a(c=2)
  # plot_vs_a(c=3)
  # plot_vs_a(c=4)
  
  plot_vs_c(ar=0.1, m=0)
  plot_vs_c(ar=0.2, m=0)
  plot_vs_c(ar=0.4, m=0)
  plot_vs_c(ar=1, m=0)
  
  plot.title(r'$V \sim$ {}'.format(V_rv) )
  plot.ylabel(r'Avg', fontsize=14)
  plot.legend()
  plot.savefig("plot_gain_pain.pdf")
  plot.gcf().clear()
  log(WARNING, "done.")

def plot_randwalk():
  num_srun = 10000*2
  V_rv = Pareto(1, 2.1) # TPareto(1, 4, 2.1) # Exp(0.1) # Exp(0.2, D=1) # TPareto(3.99, 4, 1) # DUniform(4,4)
  
  def plot_(c, m=0):
    ar_l, EJ_l, EJm_l, EJ2_l, EJ2m_l = [], [], [], [], []
    EC_l, ECm_l, ECm2_l = [], [], []
    for ar in np.linspace(0.1, 2, 10):
      ar_l.append(ar)
      X_rv = Exp(ar)
      # sim_m = sim_pain(V_rv, c, 1, X_rv, num_srun)
      # EJ_l.append(sim_m['EJ'] )
      # EJm_l.append(EJ_model(T_rv, X_rv) )
      
      # EJ2_l.append(sim_m['EJ2'] )
      # EJ2m_l.append(EJ2_model(T_rv, X_rv) )
      
      # EC_l.append(sim_m['EC'] )
      ECm_l.append(EC_model(X_rv, V_rv, m, c) )
      ECm2_l.append(EC_model_VPareto(X_rv, V_rv, m, c) )
    # plot.plot(ar_l, EJ_l, label=r'$E[J]$', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    # plot.plot(ar_l, EJm_l, label=r'$E[J]$, model', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    # plot.plot(ar_l, EJ2_l, label=r'$E[J^2]$', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    # plot.plot(ar_l, EJ2m_l, label=r'$E[J^2]$, model', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    # plot.plot(ar_l, EC_l, label=r'$E[C]$, $c= {}$'.format(c), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    plot.plot(ar_l, ECm_l, label=r'model, $E[C]$, $c= {}$'.format(c), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    plot.plot(ar_l, ECm2_l, label=r'expanded model, $E[C]$, $c= {}$'.format(c), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  plot_(c=2)
  plot_(c=3)
  plot_(c=4)
  
  plot.title(r'$V \sim$ {}'.format(V_rv) )
  plot.xlabel(r'$\lambda$', fontsize=14)
  # plot.ylabel(r'E[J]', fontsize=14)
  plot.legend()
  plot.savefig("plot_randwalk.pdf")
  plot.gcf().clear()
  log(WARNING, "done.")
"""

def plot_EG_vs_EP():
  S = Dolly()
  ES, ES2 = S.moment(1), S.moment(2)
  
  c = 1
  Smin = X_n_k(S, c+1, 1)
  ESmin, ESmin2 = Smin.moment(1), Smin.moment(2)
  
  def EG(l0, j):
    return (l0 + 1)*j*(ES - ESmin) + ar/2*j**2 * (ES2 - ESmin2)
  
  def EP(j):
    return c*ar/2*j**2 * ESmin2 # + sum(l1c)*j*ESmin
  
  

