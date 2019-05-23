from math_utils import *
from plot_utils import *
from sim_objs_lessreal import *

N, Cap = 20, 10
k = BZipf(1, 10)
R = Uniform(1, 1)
b, beta_ = 10, 3
L = Pareto(b, beta_)
a, alpha_ = 1, 3
Sl = Pareto(a, alpha_)
def alpha_gen(ro):
  return alpha_

# #####################################  Redundant-small  ######################################## #
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

def ar_for_ro0(ro, N, Cap, k, R, L, Sl):
  # log(INFO, "", ro=ro, N=N, Cap=Cap, k_mean=k.mean(), R_mean=R.mean(), L_mean=L.mean(), Sl_mean=Sl.mean() )
  return ro*N*Cap/k.mean()/R.mean()/L.mean()/Sl.mean()

def ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen):
  D = Pareto(b, beta)
  alpha = alpha_gen(ro0)
  Sl = Pareto(a, alpha)
  # log(INFO, "", alpha=alpha, alpha_gen=alpha_gen, D=D, Sl=Sl)
  return ro0*N*Cap/k.mean()/D.mean()/Sl.mean()

def Pr_kD_leq_d_pareto(k, b, beta, d):
  D = Pareto(b, beta)
  return sum([D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
  
  # def Pr_D_leq_doverk(k):
  #   if b <= d/k:
  #     return 1 - (b*k/d)**beta
  #   else:
  #     return 0
  # return sum([Pr_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )

def sim_red(k, r, B, Sl, d, red, nrun=10000):
  if d is None:
    d = 0
  S_l, C_l, S2_l, C2_l = [], [], [], []
  nsample_kB_leq_d = 0
  S_given_kB_leq_d_l, S_given_kB_g_d_l = [], []
  C_given_kB_leq_d_l, C_given_kB_g_d_l = [], []
  for i in range(nrun):
    k_ = k.sample()
    b = B.sample()
    
    if red == 'Coding':
      n = int(k_*r) if k_*b <= d else k_
      
      s_l = sorted([b*Sl.sample() for i in range(n) ] )
      s = s_l[k_-1]
      S_l.append(s)
      S2_l.append(s**2)
      
      cost = sum([min(ls, s) for ls in s_l] )
      C_l.append(cost)
      C2_l.append(cost**2)
      
      if k_*b <= d:
        nsample_kB_leq_d += 1
        S_given_kB_leq_d_l.append(s)
        C_given_kB_leq_d_l.append(cost)
      else:
        S_given_kB_g_d_l.append(s)
        C_given_kB_g_d_l.append(cost)
      
    if red == 'Rep':
      c = int(r) if k_*b <= d else 1
      s_l = sorted([b*min([Sl.sample() for j in range(c) ] ) for i in range(k_) ] )
      s = s_l[-1]
      S_l.append(s)
      S2_l.append(s**2)
      
      cost = sum([ls*c for ls in s_l] )
      C_l.append(cost)
      C2_l.append(cost**2)
  return {
    'ES': np.mean(S_l),
    'ES2': np.mean(S2_l),
    'EC': np.mean(C_l),
    'EC2': np.mean(C2_l),
    'Pr_kB_leq_d': nsample_kB_leq_d/nrun,
    'ES_given_kB_leq_d': np.mean(S_given_kB_leq_d_l),
    'ES_given_kB_g_d': np.mean(S_given_kB_g_d_l),
    'EC_given_kB_leq_d': np.mean(C_given_kB_leq_d_l),
    'EC_given_kB_g_d': np.mean(C_given_kB_g_d_l) }

def redsmall_EC_exact(k, r, b, beta, a, alpha, d=None, red=None):
  D = Pareto(b, beta)
  Sl = Pareto(a, alpha)
  if d is None:
    return k.mean()*Sl.mean()*D.mean()
  '''
  ## Block commented area gives WRONG result!
  Ek = k.mean()
  ESl = Sl.mean()
  ED = D.mean() # b/(1 - 1/beta)
  
  ED_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, i*r, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, r - 1, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  E_kD_given_kD_leq_d = sum([i*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # def Pr_kD(x):
  #   return sum([D.pdf(x/i)*k.pdf(i) for i in k.v_l] )
  # mpmath.quad(lambda x: x*Pr_kD(x), [0, d] ) \
  # E_kD_given_kD_leq_d = scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] \
  #                     / Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  
  ED_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  # EkD_given_kD_g_d = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  EkD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  
  # EkD_given_kD_g_d = (Ek*ED - scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] ) \
  #                   / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  # def Pr_kD_leq_x(x):
  #   return sum([D.cdf(x/i)*k.pdf(i) for i in k.v_l] )
  # EkD_given_kD_g_d = (Ek*ED - mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] ) )/(1 - Pr_kD_leq_d)
  
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  
  # E_kD_given_kD_leq_d = mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] )/Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  # EkD_given_kD_g_d = (Ek*ED - mpmath.quad(lambda x: 1 - Pr_kD_leq_x(x), [0, d] ) )/(1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  
  # E_kD_given_kD_leq_d = sum([mpmath.quad(lambda x: D.tail(x), [0, d/i] )*i*k.pdf(i) for i in k.v_l] ) \
  #                     / Pr_kD_leq_d if Pr_kD_leq_d != 0 else Ek*ED
  # EkD_given_kD_g_d = (Ek*ED - sum([mpmath.quad(lambda x: D.tail(x), [0, d/i] )*i*k.pdf(i) for i in k.v_l] ) ) \
  #                   / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
  
  # EkD_given_kD_g_d = (Ek*ED - E_kD_given_kD_leq_d*Pr_kD_leq_d)/(1 - Pr_kD_leq_d)
  EC_given_kD_g_d = ESl*EkD_given_kD_g_d
  
  # log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + EkD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
  # blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, EkD_given_kD_g_d=EkD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  log(INFO, "", 
    Pr_kD_leq_d=Pr_kD_leq_d,
    EC_given_kD_leq_d=EC_given_kD_leq_d,
    EC_given_kD_g_d=EC_given_kD_g_d)
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)
  '''
  ED_given_D_leq_doverk = lambda k: D.mean_given_leq_x(d/k)
  return redsmall_EC_exact(k, r, b, beta, a, alpha, d=None, red=red) \
    + sum([(EC_k_n_pareto(i, i*r, a, alpha) - i*Sl.mean())*ED_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def redsmall_EC2_exact(k, r, b, beta, a, alpha, d=None, red=None):
  D = Pareto(b, beta)
  Sl = Pareto(a, alpha)
  if d is None:
    return k.moment(2)*Sl.moment(2)*D.moment(2)
  
  ED2_given_D_leq_doverk = lambda k: moment(D, 2, given_X_leq_x=True, x=d/k)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC2_k_n_pareto(i, i*r, a, alpha)*ED2_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC2_k_c_pareto(i, r - 1, a, alpha)*ED2_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  Ek2D2_given_kD_leq_d = sum([i**2*ED2_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  Ek2D2_given_kD_g_d = (k.moment(2)*D.moment(2) - Pr_kD_leq_d*Ek2D2_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  EC_given_kD_g_d = Sl.moment(2)*Ek2D2_given_kD_g_d
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

# D ~ Pareto(b, beta), Sl ~ Pareto(a, alpha)
def redsmall_EC_model(k, r, b, beta, a, alpha, d=None, red=None):
  Ek = k.mean()
  ED = b/(1 - 1/beta)
  ESl = a/(1 - 1/alpha)
  if d is None:
    return Ek*ESl*ED
  
  def ED_given_D_leq_doverk(k):
    if b >= d/k:
      return 0
    else:
      return beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) / (1 - (b*k/d)**beta)
  if red == 'Coding':
    EC_given_kD_leq_d = sum([EC_k_n_pareto(i, i*r, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    EC_given_kD_leq_d = sum([EC_k_c_pareto(i, r - 1, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # return EC_given_kD_leq_d
  
  def ED_given_D_g_doverk(k):
    ED = b/(1 - 1/beta)
    if b > d/k:
      return ED
    else:
      return (ED - beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) ) / (b*k/d)**beta
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  E_kD_given_kD_leq_d = sum([i*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # EkD_given_kD_g_d = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  EkD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  EC_given_kD_g_d = ESl*EkD_given_kD_g_d
  # return EC_given_kD_g_d
  # log(INFO, "***", EC_given_kD_leq_d=EC_given_kD_leq_d, EC_given_kD_g_d=EC_given_kD_g_d)
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

def redsmall_EC_approx(k, r, b, beta, a, alpha, d=None, red=None):
  D = Pareto(b, beta)
  ED = b/(1 - 1/beta)
  ESl = a/(1 - 1/alpha)
  Ek = k.mean()
  if d is None:
    return Ek*ESl*ED
  
  ED_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  E_kD_given_kD_leq_d = sum([i*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  
  def ED_given_D_g_doverk(k):
    ED = b/(1 - 1/beta)
    if b > d/k:
      return ED
    else:
      return (ED - beta*b**(beta)/(beta - 1) * (b**(1-beta) - (d/k)**(1-beta) ) ) / (b*k/d)**beta
  # EkD_given_kD_g_d = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  EkD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  
  def f():
    if red == 'Coding':
      return a*r/(alpha-1) * (alpha - (1 - 1/r)**(1 - 1/alpha) )
    elif red == 'Rep':
      return a*r*alpha*r/(alpha*r - 1)
  E_kD = Ek*ED
  E_kD_ = E_kD_given_kD_leq_d*Pr_kD_leq_d + EkD_given_kD_g_d*(1 - Pr_kD_leq_d)
  # log(INFO, "E_kD= {}, E_kD_= {}".format(E_kD, E_kD_) )
  
  # return f()*E_kD_given_kD_leq_d*Pr_kD_leq_d + ESl*EkD_given_kD_g_d*(1 - Pr_kD_leq_d)
  return Ek*ESl*ED + E_kD_given_kD_leq_d*Pr_kD_leq_d*(f() - ESl)
  
  '''
  # Debugging
  # EC_given_kD_leq_d = sum([i*f()*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # EC_given_kD_leq_d = sum([EC_k_c_pareto(i, int(r)-1, a, alpha) * ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # def EC_k_c_pareto(k, c, loc, a):
  #   return k*(c+1) * a*(c+1)*loc/(a*(c+1)-1)
  # EC_given_kD_leq_d = sum([i*r * alpha*r*a/(alpha*r-1) * ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  
  EC_given_kD_leq_d = f()*E_kD_given_kD_leq_d
  
  def ED_given_D_g_doverk(k):
    if b > d/k:
      # return b/(1 - 1/beta) - d/k
      return ED
    else:
      return d/(beta-1)/k
  EkD_given_kD_g_d = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  EC_given_kD_g_d = ES*EkD_given_kD_g_d
  
  ED = b/(1 - 1/beta)
  log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + EkD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
  blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, EkD_given_kD_g_d=EkD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)
  '''

def compare_redsmall_EC_exact_approx():
  # N, Cap = 10, 100
  red = 'Coding' # 'Rep'
  r = 2
  log(INFO, "", red=red, k=k, r=r, b=b, beta_=beta_, a=a, alpha_=alpha_)
  
  L = Pareto(b, beta_)
  Sl = Pareto(a, alpha_)
  # for d in [None, *np.linspace(0.1, 10, 4), *np.linspace(100, 1000, 20) ]:
  l = a*b
  for d in [None, *np.logspace(math.log10(l), math.log10(100*l), 20) ]:
    print(">> d= {}".format(d) )
    
    sim_m = sim_red(k, r, L, Sl, d, red, nrun=2*10**4) # 2*10**4
    
    # blog(EC_exact=redsmall_EC_exact(k, r, b, beta_, a, alpha_, d, red),
    #     EC_model=redsmall_EC_model(k, r, b, beta_, a, alpha_, d, red),
    #     EC_approx=redsmall_EC_approx(k, r, b, beta_, a, alpha_, d, red),
    #     EC2_exact=redsmall_EC2_exact(k, r, b, beta_, a, alpha_, d, red),
    #     ESl=redsmall_ES(0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red),
    #     ESl2=redsmall_ES2(0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red),
    #     sim_m = sim_m)
    EC = redsmall_EC_exact(k, r, b, beta_, a, alpha_, d, red)
    EC_sim = sim_m['EC']
    print("EC= {}, EC_sim= {}".format(EC, EC_sim) )
    
    EC2 = redsmall_EC2_exact(k, r, b, beta_, a, alpha_, d, red)
    EC2_sim = sim_m['EC2']
    print("EC2= {}, EC2_sim= {}".format(EC2, EC2_sim) )
    
    ES = redsmall_ES(0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red)
    ES_sim = sim_m['ES']
    print("ES= {}, ES_sim= {}".format(ES, ES_sim) )
    log(INFO, "", sim_m=sim_m)
    
    ES2 = redsmall_ES2(0, N, Cap, k, r, b, beta_, a, alpha_gen, d, red),
    ES2_sim = sim_m['ES2']
    print("ES2= {}, ES2_sim= {}".format(ES2, ES2_sim) )
    print("\n")

def redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  def func_ro(ro):
    # return ar/N/Cap * redsmall_EC_exact(k, r, b, beta, a, alpha_gen(ro), d, red)
    return ar/N/Cap * redsmall_EC_model(k, r, b, beta, a, alpha_gen(ro), d, red)
    # return ar/N/Cap * redsmall_EC_approx(k, r, b, beta, a, alpha_gen(ro), d, red)
  
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
    u -= 0.001
  if u < l:
    log(WARNING, "u < l; u_w_max_eq= {}, max_eq= {}".format(u_w_max_eq, max_eq) )
    found_it = False
    for u in np.linspace(u_w_max_eq-0.05, u_w_max_eq+0.05, 10):
      if eq(u) > -0.01:
        found_it = True
        break
    if not found_it:
      return None
  # print("l= {}, u= {}".format(l, u) )
  try:
    ro = scipy.optimize.brentq(eq, l, u)
  except ValueError:
    return None
  # ro = scipy.optimize.newton(eq, 1)
  # ro = scipy.optimize.fixed_point(ro_, 0.5)
  # ro = scipy.optimize.fixed_point(ro_, [0.01, 0.99] )
  return ro

def redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  alpha = alpha_gen(ro)
  # log(INFO, "", ro=ro, N=N, Cap=Cap, k=k, r=r, b=b, beta=beta, a=a, alpha=alpha, d=d, red=red)
  D = Pareto(b, beta)
  if d is None:
    return D.mean()*sum([ES_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  
  '''
  ## This block commented part is WRONG!
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  # log(INFO, "", d=d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  # ED_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  # ED_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  ED_given_D_leq_doverk = lambda k: D.mean_given_leq_x(d/k)
  ED_given_D_g_doverk = lambda k: D.mean_given_g_x(d/k)
  # ED_given_D_g_doverk = lambda k: (D.mean() - ED_given_D_leq_doverk(k)*D.cdf(d/k))/D.tail(d/k)
  
  if red == 'Coding':
    ES_given_kD_leq_d = sum([ES_k_n_pareto(i, i*r, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
    # log(INFO, "",
    #   l1=[ES_k_n_pareto(i, i*r, a, alpha) for i in k.v_l],
    #   l2=[ED_given_D_leq_doverk(i) for i in k.v_l] )
  elif red == 'Rep':
    ES_given_kD_leq_d = sum([ES_k_c_pareto(i, r - 1, a, alpha)*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  ES_given_kD_g_d = sum([ES_k_n_pareto(i, i, a, alpha)*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  
  # EkD_given_kD_g_d = (Ek*ED - Pr_kD_leq_d*E_kD_given_kD_leq_d)/(1 - Pr_kD_leq_d)
  
  # ES_given_kD_g_d = sum([ES_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  # if red == 'Coding':
  #   ES_given_kD_leq_d = sum([ES_k_n_pareto(i, i*r, a, alpha)*k.pdf(i) for i in k.v_l] )
  # elif red == 'Rep':
  #   ES_given_kD_leq_d = sum([ES_k_c_pareto(i, r-1, a, alpha)*k.pdf(i) for i in k.v_l] )
  # log(INFO, "",
  #   Pr_kD_leq_d=Pr_kD_leq_d,
  #   ES_given_kD_leq_d=ES_given_kD_leq_d,
  #   ES_given_kD_g_d=ES_given_kD_g_d)
  # return ES_given_kD_leq_d*Pr_kD_leq_d + \
  #       ES_given_kD_g_d*(1 - Pr_kD_leq_d)
  '''
  ED_given_D_leq_doverk = lambda k: D.mean_given_leq_x(d/k)
  return redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=red) \
    + sum([(ES_k_n_pareto(i, i*r, a, alpha) - ES_k_n_pareto(i, i, a, alpha) )*ED_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def redsmall_ES2(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  alpha = alpha_gen(ro)
  D = Pareto(b, beta)
  if d is None:
    return D.moment(2)*sum([ES2_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  '''
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  
  # ED_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
  # ED_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
  ED2_given_D_leq_doverk = lambda k: moment(D, 2, given_X_leq_x=True, x=d/k)
  ED2_given_D_g_doverk = lambda k: moment(D, 2, given_X_leq_x=False, x=d/k)
  
  ESl2_given_kD_g_d = sum([ES2_k_n_pareto(i, i, a, alpha)*ED2_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
  if red == 'Coding':
    ESl2_given_kD_leq_d = sum([ES2_k_n_pareto(i, i*r, a, alpha)*ED2_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    ESl2_given_kD_leq_d = sum([ES2_k_n_pareto(i, r - 1, a, alpha)*ED2_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
  # ESl2_given_kD_g_d = sum([ES2_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  # if red == 'Coding':
  #   ESl2_given_kD_leq_d = sum([ES2_k_n_pareto(i, i*r, a, alpha)*k.pdf(i) for i in k.v_l] )
  # elif red == 'Rep':
  #   ESl2_given_kD_leq_d = sum([ES2_k_c_pareto(i, r-1, a, alpha)*k.pdf(i) for i in k.v_l] )
  return ESl2_given_kD_leq_d*Pr_kD_leq_d + \
         ESl2_given_kD_g_d*(1 - Pr_kD_leq_d)
  '''
  ED2_given_D_leq_doverk = lambda k: moment(D, 2, given_X_leq_x=True, x=d/k)
  return redsmall_ES2(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=red) \
    + sum([(ES2_k_n_pareto(i, i*r, a, alpha) - ES2_k_n_pareto(i, i, a, alpha) )*ED2_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red):
  '''Using the result for M/M/c to approximate E[T] in M/G/c.
     [https://en.wikipedia.org/wiki/M/G/k_queue]
  '''
  ar = ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None, None
  alpha = alpha_gen(ro)
  
  ES = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  ES2 = redsmall_ES2(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  EC = redsmall_EC_exact(k, r, b, beta, a, alpha, d, red)
  # B = Pareto(b, beta)
  # Sl = Pareto(a, alpha)
  # sim_m = sim_red(k, r, B, Sl, d, red, nrun=2*10**4)
  # ES, ES2, EC = sim_m['ES'], sim_m['ES2'], sim_m['EC']
  log(INFO, "", ES=ES, ES2=ES2, EC=EC)
  
  # log(INFO, "ar*EC/(N*Cap)= {}".format(ar*EC/(N*Cap) ) )
  EW, Prqing = MGc_EW_Prqing(ar, N*Cap*ES/EC, ES, ES2)
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  log(INFO, "d= {}, ro= {}".format(d, ro) )
  return round(ET, 2), round(EW, 2), round(Prqing, 2)

def redsmall_optimal_d(ro0, N, Cap, k, r, b, beta, a, alpha_gen, red, max_d=None):
  '''
  ET_base = 1.1*redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)[0]
  if ET_base is None:
    return 0
  
  L = Pareto(b, beta)
  db, de = 0, 10**3*L.mean()
  ET = 0
  while de - db > 10: # ET < ET_base:
    d = (de + db)/2
    ET = redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)[0]
    if ET is None:
      return db
    
    if ET < ET_base:
      db = d
    else:
      de = d
  return db
  '''
  func = lambda d: redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)[0]
  
  if max_d is None:
    d = 0
    ET_d0 = redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)[0]
    print(">> ET_d0= {}".format(ET_d0) )
    
    L = Pareto(b, beta)
    EL = L.mean()
    # print("EL= {}".format(EL) )
    l = 0 # 10
    u = 10**3*EL
    while u - l > 10:
      d = (l + u)/2
      ET = redsmall_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)[0]
      if ET is None or ET > 1.5*ET_d0:
        u = d
      else:
        l = d
    max_d = (l + u)/2
  # log(INFO, "max_d= {}".format(max_d) )
  
  r = scipy.optimize.minimize_scalar(func, bounds=(0, max_d), method='bounded')
  log(INFO, "r= {}".format(r) )
  return round(r.x, 1)

def redsmall_approx_ET_EW_Prqing_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red):
  ar = ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None, None
  alpha = alpha_gen(ro)
  
  L = Pareto(b, beta)
  ES = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  EW = 1/ar * ro**2/(1 - ro)
  
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  log(INFO, "d= {}, ro= {}".format(d, ro) )
  return round(ET, 2), round(EW, 2), round(ro, 2)

def redsmall_ET_EW(ro0, EW0, N, Cap, k, r, b, beta, a, alpha_gen, d, red, K=None):
  '''Using (1) to approximate E[T] in M/G/c with heavy tailed job sizes.
     [Konstantinos Psounis, "Systems with Multiple Servers under Heavy-tailed Workloads"
      http://www-bcf.usc.edu/~kpsounis/EE650/Readlist07/Papers07/multiserver.pdf] '''
  if K is None:
    alpha0 = alpha_gen(ro0)
    K = EW0/(ro0/(1-ro0)*redsmall_EC2_exact(k, r, b, beta, a, alpha0)/redsmall_EC_exact(k, r, b, beta, a, alpha0) )
  
  ar = ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None
  alpha = alpha_gen(ro)
  
  L = Pareto(b, beta) # Take D as the lifetime; D = LR, R = 1
  ES = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  ES2 = redsmall_ES2(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  print("ES= {}, ES2= {}".format(ES, ES2) )
  EC = redsmall_EC_exact(k, r, b, beta, a, alpha, d, red)
  EC2 = redsmall_EC2_exact(k, r, b, beta, a, alpha, d, red)
  print("EC= {}, EC2= {}".format(EC, EC2) )
  def Pr_blocking(ar, ro):
    Pr_shortjob = L.cdf(5*L.mean() ) # 0.9 # 1/2
    long_jlifetime = ES + math.sqrt((ES2 - ES**2)*Pr_shortjob/(1-Pr_shortjob) ) # 10*L.mean()
    narr_atleast_forblocking = (1 - L.cdf(long_jlifetime)*ro)*N*Cap / (EC/ES) - 1
    ar_long = ar*L.tail(long_jlifetime)*long_jlifetime
    # narr_atleast_forblocking = (1 - ro)*N*Cap/k.moment(1) - 1
    # ar_long = ar*ES
    
    blog(Pr_shortjob=Pr_shortjob, long_jlifetime=long_jlifetime, narr_atleast_forblocking=narr_atleast_forblocking, ar_long=ar_long)
    return 1 - math.exp(-ar_long)*sum([ar_long**i/math.factorial(i) for i in range(int(narr_atleast_forblocking)+1) ] )
  
  pblocking = K # Pr_blocking(ar, ro)
  EW = ro/(1-ro)*EC2/EC * pblocking
  
  ET = ES + EW
  log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}, pblocking= {}".format(d, ro, ES, EW, ET, pblocking) )
  return ET, EW

def plot_ET():
  N, Cap = 10, 100
  b, beta = 10, 3
  a, alpha = 1, 3
  # D = Pareto(b, beta)
  # S = Pareto(a, alpha)
  k = BZipf(1, 10)
  def alpha_gen(ro):
    return alpha
  ro0, EW0 = 0.55, 20
  
  fontsize = 14
  def plot_(red, r):
    log(INFO, "red= {}, r= {}".format(red, r) )
    d_l, ET_l = [], []
    l = a*b # D.mean()*S.mean()
    for d in np.logspace(math.log10(l), math.log10(100*l), 20):
      print(">> d= {}".format(d) )
      ET, EW = redsmall_ET_EW(ro0, EW0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
      print("ET= {}".format(ET) )
      d_l.append(d)
      ET_l.append(ET)
    plot.plot(d_l, ET_l, label='w/ {}, r={}'.format(red, r), c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=1)
  
  plot_('Rep', 2)
  plot_('Coding', 2)
  plot_('Rep', 3)
  plot_('Coding', 3)
  
  prettify(plot.gca() )
  plot.legend()
  plot.xscale('log')
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('E[T]', fontsize=fontsize)
  
  plot.title(r'$N= {}$, $C= {}$, $k \sim$ {}'.format(N, Cap, k) + '\n' + r'$\rho_0= {}$, $E[W_0]= {}$, $b= {}$, $\beta= {}$, $a= {}$, $\alpha= {}$'.format(ro0, EW0, b, beta, a, alpha) )
  plot.gcf().set_size_inches(5, 5)
  plot.savefig('plot_ET.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

def redsmall_r_max_wo_exceeding_EC0(N, Cap, k, b, beta, a, alpha_gen, red):
  EL = Pareto(b, beta).mean()
  d = 10**3*EL
  EC_func = lambda r: redsmall_EC_exact(k, r, b, beta, a, alpha, d, red)
  
  EC0 = EC_func(r=1)
  l, u = 1, 10
  while u - l > 0.001:
    r = (l + u)/2
    EC = EC_func(r)
    if EC is None or EC > EC0:
      u = r
    else:
      l = r
    print("l= {}, u= {}".format(l, u) )
  r = (l + u)/2  
  log(INFO, "r= {}".format(r) )
  return r

# #######################################  Relaunch-Delta  ####################################### #
## d: relaunch time
def relaunch_ES_pareto(k, a, alpha, w):
  d = a*w
  if d <= a:
    return d + ES_k_n_pareto(k, k, a, alpha)
  
  q = (d > a)*(1 - (a/d)**alpha)
  L = a*G(1-1/alpha)*G(k+1)/G(k+1-1/alpha) if k < 170 else a*(k+1)**(1/alpha) * G(1-1/alpha)
  return d*(1-q**k) + L*(1 + (a/d-1)*I(1-q,1-1/alpha,k) )

def relaunch_ES2_pareto(k, a, alpha, w):
  # log(INFO, "", k=k, a=a, alpha=alpha, w=w)
  d = a*w
  if d <= a:
    return d**2 + 2*d*ES_k_n_pareto(k, k, a, alpha) + ES2_k_n_pareto(k, k, a, alpha)
  
  q = (d > a)*(1 - (a/d)**alpha)
  E_ = lambda r: d**2 + 2*d*ES_k_n_pareto(k-r, k-r, a, alpha) + \
                 ES2_k_n_pareto(k-r, k-r, a, alpha) - ES2_k_n_pareto(k-r, k-r, d, alpha)
  return ES2_k_n_pareto(k, k, a, alpha) + \
         sum([E_(r) * binom_(k, r) * q**r * (1-q)**(k-r) for r in range(k) ] )

def relaunch_derived_ES2_pareto(k, a, alpha, w):
  d = a*w
  if d <= a:
    return d**2 + 2*d*ES_k_n_pareto(k, k, a, alpha) + ES2_k_n_pareto(k, k, a, alpha)
  
  q = (d > a)*(1 - (a/d)**alpha)
  E_ = lambda r: ES_k_n_pareto(k-r, k-r, a, alpha)
  return ES2_k_n_pareto(k, k, a, alpha) + \
         d**2*(1 - q**k) + \
         2*d*a*G(1-1/alpha)*G(k+1)/G(k+1-1/alpha) * (1-q)**(1/alpha)*I(1-q, 1-1/alpha, k) + \
         (a**2 - d**2)*G(1-2/alpha)*G(k+1)/G(k+1-2/alpha) * (1-q)**(2/alpha)*I(1-q, 1-2/alpha, k)

def relaunch_ES2_exact_vs_approx():
  for k in range(5, 10):
    print("\n>> k= {}".format(k) )
    for w in np.linspace(1, 4, 20):
      print("w= {}".format(w) )
      relaunch_ES2 = relaunch_ES2_pareto(k, a, alpha, w)
      relaunch_derived_ES2 = relaunch_derived_ES2_pareto(k, a, alpha, w)
      print("relaunch_ES2= {}, relaunch_derived_ES2= {}".format(relaunch_ES2, relaunch_derived_ES2) )

def relaunch_EC_pareto(k, a, alpha, w):
  d = a*w
  if d == 0:
    return k*a/(alpha-1) * (alpha - G(k)*G(1-1/a)/G(k+1-1/a) )
  
  q = (d > a)*(1 - (a/d)**alpha)
  if d <= a:
    return k*d + relaunch_EC_pareto(k, a, alpha, 0)
  else:
    return alpha/(alpha-1)*(k*(1-q)*(a-d) + k*a)

def relaunch_ES(k, b, beta, a, alpha, w):
  EL = Pareto(b, beta).mean()
  ES_wrt_k = lambda k: EL*relaunch_ES_pareto(k, a, alpha, w)
  return sum([ES_wrt_k(i)*k.pdf(i) for i in k.v_l] )

def relaunch_ES2(k, b, beta, a, alpha, w):
  EL2 = Pareto(b, beta).moment(2)
  ES2_wrt_k = lambda k: EL2*relaunch_ES2_pareto(k, a, alpha, w)
  return sum([ES2_wrt_k(i)*k.pdf(i) for i in k.v_l] )

def relaunch_EC(k, b, beta, a, alpha, w):
  EL = Pareto(b, beta).mean()
  EC_wrt_k = lambda k: EL*relaunch_EC_pareto(k, a, alpha, w)
  return sum([EC_wrt_k(i)*k.pdf(i) for i in k.v_l] )

def relaunch_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta, a, alpha, w):
  '''Using the result for M/M/c to approximate E[T] in M/G/c.
     [https://en.wikipedia.org/wiki/M/G/k_queue]
  '''
  ar = ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, lambda ro: alpha)
  
  ES = relaunch_ES(k, b, beta, a, alpha, w)
  ES2 = relaunch_ES2(k, b, beta, a, alpha, w)
  EC = relaunch_EC(k, b, beta, a, alpha, w)
  
  # log(INFO, "ar*EC/(N*Cap)= {}".format(ar*EC/(N*Cap) ) )
  EW, Prqing = MGc_EW_Prqing(ar, N*Cap*ES/EC, ES, ES2)
  ET = ES + EW
  ro = ar*EC/N/Cap
  log(INFO, "w= {}, ro= {}".format(w, ro) )
  return round(ET, 2), round(EW, 2), round(Prqing, 2)

def relaunch_approx_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta, a, alpha, w):
  ar = ar_for_ro0_pareto(ro0, N, Cap, k, b, beta, a, lambda ro: alpha)
  
  ES = relaunch_ES(k, b, beta, a, alpha, w)
  EC = relaunch_EC(k, b, beta, a, alpha, w)
  
  ro = ar*EC/N/Cap
  EW = 1/ar * ro**2/(1 - ro)
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  log(INFO, "w= {}, ro= {}".format(w, ro) )
  return round(ET, 2), round(EW, 2), round(ro, 2)

def relaunch_opt_w_using_ES(k, b, a, alpha):
  ES_w1 = b*relaunch_ES_pareto(k, a, alpha, w=1)
  
  l, u = 1, 10
  while u - l > 0.05:
    w = (l + u)/2
    ES = b*relaunch_ES_pareto(k, a, alpha, w)
    if ES is None or ES > 1.2*ES_w1:
      u = w
    else:
      l = w
  max_w = (l + u)/2
  # log(INFO, "max_d= {}".format(max_d) )
  
  func = lambda w: relaunch_ES_pareto(k, a, alpha, w)
  r = scipy.optimize.minimize_scalar(func, bounds=(1, max_w), method='bounded')
  return round(r.x, 1)

def relaunch_opt_w_using_ET(ro0, N, Cap, k, b, beta, a, alpha):
  ET_w1 = relaunch_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta, a, alpha, w=1)[0]
  
  l, u = 1, 10
  while u - l > 0.05:
    w = (l + u)/2
    ET = relaunch_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta, a, alpha, w)[0]
    if ET is None or ET > 1.2*ET_w1:
      u = w
    else:
      l = w
  max_w = (l + u)/2
  # log(INFO, "max_d= {}".format(max_d) )
  
  func = lambda w: relaunch_ET_EW_Prqing_wMGc(ro0, N, Cap, k, b, beta, a, alpha, w)[0]
  r = scipy.optimize.minimize_scalar(func, bounds=(1, max_w), method='bounded')
  return round(r.x, 1)

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
  
  ar = ar_for_ro0_pareto(1/2, N, Cap, k, b, beta, a, alpha_gen)
  print("ar= {}".format(ar) )
  
  d = None
  ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d)
  print("ro= {}".format(ro) )
  Esl = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d)
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
    ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    Esl = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
    blog(ro=ro, Esl=Esl)
    ro_wrep_l.append(ro)
    Esl_wrep_l.append(Esl)
    
    red = 'Coding'
    ro = redsmall_ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    Esl = redsmall_ES(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
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

if __name__ == "__main__":
  # plot_slowdown()
  compare_redsmall_EC_exact_approx()
  # plot_ro_Esl()
  
  # red = 'Coding'
  # r = redsmall_r_max_wo_exceeding_EC0(N, Cap, k, b, beta_, a, alpha_gen, red)
  # print("r= {}".format(r) )
  
  # relaunch_ES2_exact_vs_approx()
  # print("binom_(5, 1)= {}".format(binom_(5, 1) ) )
  
  ro0 = 0.5
  # opt_w = relaunch_opt_w_using_ET(ro0, N, Cap, k, b, beta_, a, alpha_)
  # opt_w = relaunch_opt_w_using_ES(k=5, b=1, a=a, alpha=alpha_)
  # log(INFO, "opt_w= {}".format(opt_w) )
