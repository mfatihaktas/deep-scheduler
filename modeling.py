from math_utils import *
from plot_utils import *
from sim_objs_lessreal import *

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

def sim_red(k, r, L, Sl, d, red, nrun=10000):
  if d is None:
    d = 0
  T_l, C_l, T2_l, C2_l = [], [], [], []
  for i in range(nrun):
    k_ = k.sample()
    L_ = L.sample()
    
    if red == 'Coding':
      n = int(k_*r) if k_*L_ <= d else k_
      LS_l = sorted([L_*Sl.sample() for i in range(n) ] )
      T = LS_l[k_-1]
      T_l.append(T)
      T2_l.append(T**2)
      
      C = sum([min(ls, ET) for ls in LS_l] )
      C_l.append(C)
      C2_l.append(C**2)
    if red == 'Rep':
      c = int(r) if k_*L_ <= d else 1
      LS_l = sorted([L_*min([Sl.sample() for j in range(c) ] ) for i in range(k_) ] )
      T = LS_l[-1]
      T_l.append(T)
      T2_l.append(T**2)
      
      C = sum([ls*c for ls in LS_l] )
      C_l.append(C)
      C2_l.append(C**2)
  return {
    'ET': np.mean(T_l),
    'ET2': np.mean(T2_l),
    'EC': np.mean(C_l),
    'EC2': np.mean(C2_l) }

def EC_exact_pareto(k, r, b, beta, a, alpha, d=None, red=None):
  D = Pareto(b, beta)
  Sl = Pareto(a, alpha)
  if d is None:
    return k.mean()*Sl.mean()*D.mean()
  
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
  
  return EC_given_kD_leq_d*Pr_kD_leq_d + \
         EC_given_kD_g_d*(1 - Pr_kD_leq_d)

def EC2_exact_pareto(k, r, b, beta, a, alpha, d=None, red=None):
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
def EC_model_pareto(k, r, b, beta, a, alpha, d=None, red=None):
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

def EC_approx_pareto(k, r, b, beta, a, alpha, d=None, red=None):
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

def compare_EC_exact_approx():
  # N, Cap = 10, 100
  red = 'Rep' # 'Coding'
  k = BZipf(1, 10)
  r = 2
  b, beta = 10, 3 # 1.1
  a, alpha = 1, 3 # 2
  log(INFO, "", red=red, k=k, r=r, b=b, beta=beta, a=a, alpha=alpha)
  
  L = Pareto(b, beta)
  Sl = Pareto(a, alpha)
  # for d in [None, *np.linspace(0.1, 10, 4), *np.linspace(100, 1000, 20) ]:
  l = a*b
  for d in [None, *np.logspace(math.log10(l), math.log10(100*l), 20) ]:
    print(">> d= {}".format(d) )
    
    sim_m = sim_red(k, r, L, Sl, d, red, nrun=2*10**4)
    
    blog(EC_exact=EC_exact_pareto(k, r, b, beta, a, alpha, d, red),
         EC_model=EC_model_pareto(k, r, b, beta, a, alpha, d, red),
         EC_approx=EC_approx_pareto(k, r, b, beta, a, alpha, d, red),
         EC2_exact=EC2_exact_pareto(k, r, b, beta, a, alpha, d, red),
         sim_EC = sim_m['EC'] )

def ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  def func_ro(ro):
    # return ar/N/Cap * EC_exact_pareto(k, r, b, beta, a, alpha_gen(ro), d, red)
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
    u -= 0.001
  if u < l:
    print("u < l; u_w_max_eq= {}, max_eq= {}".format(u_w_max_eq, max_eq) )
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

def ar_for_ro_pareto(ro, N, Cap, k, b, beta, a, alpha_gen):
  D = Pareto(b, beta)
  Sl = Pareto(a, alpha_gen(ro) )
  return ro*N*Cap/k.mean()/D.mean()/Sl.mean()

def ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  alpha = alpha_gen(ro)
  if d is None:
    return sum([ET_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  # log(INFO, "", d=d, Pr_kD_leq_d=Pr_kD_leq_d)
  
  # S = Pareto(a, alpha_gen(ro) )
  # ES_given_kD_g_d = sum([X_n_k(S, i, i).mean()*k.pdf(i) for i in k.v_l] )
  # ES_given_kD_leq_d = sum([X_n_k(S, int(i*r), i).mean()*k.pdf(i) for i in k.v_l] )
  # return ES_given_kD_leq_d*Pr_kD_leq_d + \
  #       ES_given_kD_g_d*(1 - Pr_kD_leq_d)
  
  ES_given_kD_g_d = sum([ET_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  if red == 'Coding':
    ES_given_kD_leq_d = sum([ET_k_n_pareto(i, i*r, a, alpha)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    ES_given_kD_leq_d = sum([ET_k_c_pareto(i, r-1, a, alpha)*k.pdf(i) for i in k.v_l] )
  return ES_given_kD_leq_d*Pr_kD_leq_d + \
         ES_given_kD_g_d*(1 - Pr_kD_leq_d)

def ESl2_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d=None, red=None):
  alpha = alpha_gen(ro)
  if d is None:
    return sum([ET_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  
  Pr_kD_leq_d = Pr_kD_leq_d_pareto(k, b, beta, d)
  
  ESl2_given_kD_g_d = sum([ET2_k_n_pareto(i, i, a, alpha)*k.pdf(i) for i in k.v_l] )
  if red == 'Coding':
    ESl2_given_kD_leq_d = sum([ET2_k_n_pareto(i, i*r, a, alpha)*k.pdf(i) for i in k.v_l] )
  elif red == 'Rep':
    ESl2_given_kD_leq_d = sum([ET2_k_c_pareto(i, r-1, a, alpha)*k.pdf(i) for i in k.v_l] )
  return ESl2_given_kD_leq_d*Pr_kD_leq_d + \
         ESl2_given_kD_g_d*(1 - Pr_kD_leq_d)

def ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red):
  '''Using the result for M/M/c to approximate E[T] in M/G/c.
     [https://en.wikipedia.org/wiki/M/G/k_queue]
  '''
  ar = ar_for_ro_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None, None
  alpha = alpha_gen(ro)
  
  def MGc_EW_Prqing(ar, c, EX, EX2):
    def MMc_EW_Prqing(ar, EX, c):
      ro = ar*EX/c
      ro_ = c*ro
      # Prqing = 1/(1 + (1-ro)*G(c+1)/ro_**c * sum([ro_**i/G(i+1) for i in range(c) ] ) )
      c_times_ro__power_c = math.exp(c*math.log(c*ro) )
      Prqing = 1/(1 + (1-ro) * math.exp(ro_)*G(c, ro_, 'upper')/c_times_ro__power_c)
      
      # EN = ro/(1-ro)*Prqing + c*ro
      # log(INFO, "ro= {}, Prqing= {}".format(ro, Prqing) )
      return Prqing/(c/EX - ar), Prqing
    # CoeffVar = math.sqrt(EX2 - EX**2)/EX
    # return (1 + CoeffVar**2)/2 * MMc_EW_Prqing(ar, EX, c)
    MMc_EW, MMc_Prqing = MMc_EW_Prqing(ar, EX, c)
    return (1 + (EX2 - EX**2)/EX**2)/2 * MMc_EW, MMc_Prqing
  L = Pareto(b, beta)
  ES = L.mean()*ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  ES2 = L.moment(2)*ESl2_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  EC = EC_exact_pareto(k, r, b, beta, a, alpha, d, red)
  
  # log(INFO, "ar*EC/(N*Cap)= {}".format(ar*EC/(N*Cap) ) )
  EW, Prqing = MGc_EW_Prqing(ar, N*Cap*ES/EC, ES, ES2)
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  return round(ET, 2), round(EW, 2), round(Prqing, 2)

def approx_ET_EW_Prqing_pareto_wMGc(ro0, N, Cap, k, r, b, beta, a, alpha_gen, d, red):
  ar = ar_for_ro_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None, None
  alpha = alpha_gen(ro)
  
  L = Pareto(b, beta)
  ES = L.mean()*ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  EW = 1/ar * ro**2/(1 - ro)
  
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  return round(ET, 2), round(EW, 2), round(ro, 2)

def ET_EW_pareto(ro0, EW0, N, Cap, k, r, b, beta, a, alpha_gen, d, red, K=None):
  '''Using (1) to approximate E[T] in M/G/c with heavy tailed job sizes.
     [Konstantinos Psounis, "Systems with Multiple Servers under Heavy-tailed Workloads"
      http://www-bcf.usc.edu/~kpsounis/EE650/Readlist07/Papers07/multiserver.pdf] '''
  if K is None:
    alpha0 = alpha_gen(ro0)
    K = EW0/(ro0/(1-ro0)*EC2_exact_pareto(k, r, b, beta, a, alpha0)/EC_exact_pareto(k, r, b, beta, a, alpha0) )
  
  ar = ar_for_ro_pareto(ro0, N, Cap, k, b, beta, a, alpha_gen)
  ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  if ro is None:
    return None, None
  alpha = alpha_gen(ro)
  
  L = Pareto(b, beta) # Take D as the lifetime; D = LR, R = 1
  ES = L.mean()*ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  ES2 = L.moment(2)*ESl2_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
  print("ES= {}, ES2= {}".format(ES, ES2) )
  EC = EC_exact_pareto(k, r, b, beta, a, alpha, d, red)
  EC2 = EC2_exact_pareto(k, r, b, beta, a, alpha, d, red)
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
      ET, EW = ET_EW_pareto(ro0, EW0, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
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
  Esl = ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d)
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
    Esl = ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
    blog(ro=ro, Esl=Esl)
    ro_wrep_l.append(ro)
    Esl_wrep_l.append(Esl)
    
    red = 'Coding'
    ro = ro_pareto(ar, N, Cap, k, r, b, beta, a, alpha_gen, d, red)
    Esl = ESl_pareto(ro, N, Cap, k, r, b, beta, a, alpha_gen, d, red) if ro is not None else None
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
  compare_EC_exact_approx()
  # plot_ro_Esl()
