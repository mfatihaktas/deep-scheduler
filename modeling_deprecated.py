import numpy as np
import mpmath

from rvs import *

'''
Kubernetes architecture; master assigning jobs to distributed workers.
Average cluster load = E[ro] = ar/N/Cap * E[D x S]
where
ar: Arrival rate of jobs
N: Number of workers
Cap: Capacity of each worker
D: Total demand of a job; lifetime x resource demand
S: Slowdown experienced by each task

S is assumed to depend only on ro.
Redundancy is introduced for jobs with D < D0.
'''
def E_slowdown(ar, N, Cap, D, S_gen, D0=None):
  def ro_(ro):
    if D0 is not None:
      ED_given_D_leq_D0 = mean(D, given_X_leq_x=True, x=D0)
      S = S_gen(ro)
      S_2_1 = X_n_k(S, 2, 1)
      ES_given_D_leq_D0 = S_2_1.moment(1)
      
      ED_given_D_g_D0 = mean(D, given_X_leq_x=False, x=D0)
      ES_given_D_g_D0 = S.moment(1)
      # blog(E_S_2_1=ES_given_D_leq_D0, ES=ES_given_D_g_D0)
      
      Pr_D_leq_D0 = D.cdf(D0)
      
      EA = 2*ED_given_D_leq_D0*ES_given_D_leq_D0*Pr_D_leq_D0 + \
           ED_given_D_g_D0*ES_given_D_g_D0*(1 - Pr_D_leq_D0)
    else:
      EA = D.mean()*S_gen(ro).mean()
    return ar/N/Cap * EA
  
  eq = lambda ro: ro - ro_(ro)
  l, u = 0.0001, 1
  roots = scipy.optimize.brentq(eq, l, u)
  print("roots= {}".format(roots) )
  # for x in np.linspace(l, u, 40):
  #   print("eq({})= {}".format(x, eq(x) ) )
  
  S = S_gen(roots)
  return S.mean()

def arrival_rate_ub(N, Cap, D, S_gen):
  return N*Cap/D.mean()/S_gen(1).mean()  

def plot_slowdown():
  N, Cap = 10, 100
  D = TPareto(1, 10000, 2) # Pareto(10, 2)
  
  # S_gen = lambda ro: TPareto(1, 40, 2/ro)
  def S_gen(ro):
    a = 1.5 - math.sqrt(ro) # 2 - ro
    return TPareto(1, 1000, a)
  ar_ub = arrival_rate_ub(N, Cap, D, S_gen)
  print("ar_ub= {}".format(ar_ub) )
  
  # D0 = 2*D.mean()
  # for ar in np.linspace(ar_ub/20, ar_ub, 10):
  #   E_sl = E_slowdown(ar, N, Cap, D, S_gen)
  #   E_sl_wred = E_slowdown(ar, N, Cap, D, S_gen, D0)
  #   print("ar= {}, E_sl= {}, E_sl_wred= {}".format(ar, E_sl, E_sl_wred) )
  
  ar = 1/2*ar_ub # 2/3
  l, u = 1.1*D.l_l, 0.95*D.u_l
  # for D0 in np.linspace(l, u, 10):
  for D0 in np.logspace(math.log10(l), math.log10(u), 10):
    print("D0= {}".format(D0) )
    E_sl = E_slowdown(ar, N, Cap, D, S_gen)
    E_sl_wred = E_slowdown(ar, N, Cap, D, S_gen, D0)
    blog(E_sl=E_sl, E_sl_wred=E_sl_wred)
  

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
    Pr_kD_leq_d = sum([D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
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
      EC_given_kD_leq_d = sum([E_cumS(i)*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
      
      ## kD > d
      EC_given_kD_g_d = ES*sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
      
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
  E_S_given_kD_g_d = sum([X_n_k(S, i, i).mean()*k.pdf(i) for i in k.v_l] )
  if d is not None:
    E_S_given_kD_leq_d = sum([X_n_k(S, i+r, i).mean()*k.pdf(i) for i in k.v_l] )
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
  plot.xlabel('d', fontsize=fontsize)
  plot.ylabel('Average slowdown', fontsize=fontsize)
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
    
    E_kD_given_kD_leq_d = sum([i*E_D_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
    Pr_kD_leq_d = sum([D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
    E_kD_given_kD_g_d = sum([i*E_D_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
    
    E_kD = k.mean()*D.mean()
    E_kD_totalsum = Pr_kD_leq_d*E_kD_given_kD_leq_d + (1 - Pr_kD_leq_d)*E_kD_given_kD_g_d
    log(INFO, "d= {}".format(d), E_kD=E_kD, E_kD_totalsum=E_kD_totalsum)
  
  for d in np.linspace(10, 1000, 10):
    do_for(d)

def check_MGc_assumption():
  # N, Cap = 10, 1
  N_times_Cap = 100
  r = 1
  L = Exp(1, 1)
  S = DUniform(1, 1)
  sinfo_m['njob'] = 2000*10
  sching_m = {'type': 'plain', 'r': r}
  blog(N_times_Cap=N_times_Cap, sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  def run(ro, N, k, R, L, S, r=1):
    Cap = int(N_times_Cap/N)
    print("\n")
    log(INFO, "ro= {}, N= {}, Cap= {}, k= {}, R= {}, L= {}, S= {}, r= {}".format(ro, N, Cap, k, R, L, S, r) )
    
    ar = round(ar_for_ro(ro, N, Cap, k, R, L, S), 2)
    sinfo_m.update({
      'nworker': N, 'wcap': Cap, 'ar': ar,
      'k_rv': k,
      'reqed_rv': R,
      'lifetime_rv': L,
      'straggle_m': {'slowdown': lambda load: S.sample() } } )
    sching_m['r'] = r
    sim_m = sim(sinfo_m, mapping_m, sching_m, "N{}_C{}".format(N, Cap) )
    blog(sim_m=sim_m)
    
    # c = int(N*Cap/R.mean() ) # N*Cap
    # print("c= {}".format(c) )
    # EW = EW_MGc(ar, L, c)
    # print("M/G/c_EW= {}".format(EW) )
    return {
      'ar': ar,
      'EW': sim_m['waittime_mean'],
      'pblocking': sim_m['frac_jobs_waited_inq'],
      'EW_givenqed': sim_m['waittime_givenqed_mean'] }
  
  def test(ro, R=DUniform(1, 1) ):
    print("---------------")
    run(ro, 1, k, R, L, S)
    # run(ro, 2, k, R, L, S)
    # run(ro, 5, k, R, L, S)
    # run(ro, 10, k, R, L, S)
  
  def check_EW_scaling_wrt_ro(N, R):
    log(INFO, "", N=N, R=R)
    
    # '''
    ro_l, EW_l = [], []
    for ro in np.linspace(0.1, 0.9, 9):
      ro = round(ro, 2)
      ar, EW, pblocking = run(ro, N, k, R, L, S)
      print("ro= {}, EW= {}".format(ro, EW) )
      
      ro_l.append(ro)
      EW_l.append(EW)
    blog(ro_l=ro_l, EW_l=EW_l)
    # '''
    
    # ro_l= [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # EW_l= [0.00025548087470978202, 0.00056689800613990546, 0.00089200542402208672, 0.0012637166320921696, 0.0017178514022176334, 0.0021802843452227629, 0.002912705863562876, 0.0061096923858674568, 0.043253547318583753]
    print("ratio = EW/(ro/(1-ro))")
    for i, EW in enumerate(EW_l):
      ro = ro_l[i]
      ratio = EW/(ro/(1-ro) )
      print("ro= {}, ratio= {}".format(ro, ratio) )
    log(INFO, "done.")
  
  def check_EW_scaling_wrt_EL2_over_EL(N, R, ro):
    log(INFO, "", N=N, R=R, ro=ro)
    
    EL2_over_EL_l, EW_l = [], []
    for mu in np.linspace(0.1, 1, 10):
      L = Exp(mu, 1)
      EL2_over_EL = round(L.moment(2)/L.moment(1), 2)
      ar, EW, pblocking = run(ro, N, k, R, L, S)
      print("EL2_over_EL= {}, EW= {}".format(EL2_over_EL, EW) )
      
      EL2_over_EL_l.append(EL2_over_EL)
      EW_l.append(EW)
    blog(EL2_over_EL_l=EL2_over_EL_l, EW_l=EW_l)
    # '''
    
    print("ratio = EW/(EL2/EL)")
    for i, EW in enumerate(EW_l):
      EL2_over_EL = EL2_over_EL_l[i]
      ratio = EW/EL2_over_EL
      print("EL2_over_EL= {}, ratio= {}".format(EL2_over_EL, ratio) )
    log(INFO, "done.")
  
  def check_EW_scaling_wrt_ER2_over_ER(N, L, ro):
    log(INFO, "", N=N, L=L, ro=ro)
    
    ER2_over_ER_l, EW_l = [], []
    for u in np.linspace(0.1, 1, 10):
      R = Uniform(0.1, u)
      ER2_over_ER = round(R.moment(2)/R.moment(1), 2)
      ar, EW, pblocking = run(ro, N, k, R, L, S)
      print("ER2_over_ER= {}, EW= {}".format(ER2_over_ER, EW) )
      
      ER2_over_ER_l.append(ER2_over_ER)
      EW_l.append(EW)
    blog(ER2_over_ER_l=ER2_over_ER_l, EW_l=EW_l)
    
    print("ratio = EW/(ER2/ER)")
    for i, EW in enumerate(EW_l):
      ER2_over_ER = ER2_over_ER_l[i]
      ratio = EW/ER2_over_ER
      print("ER2_over_ER= {}, ratio= {}".format(ER2_over_ER, ratio) )
    log(INFO, "done.")
  
  def check_EW_scaling_wrt_Ek2_over_Ek(N, R, L, ro):
    log(INFO, "", N=N, R=R, L=L, ro=ro)
    
    Ek2_over_Ek_l, EW_l = [], []
    for u in range(1, 10):
      k = DUniform(1, u)
      Ek2_over_Ek = round(k.moment(2)/k.moment(1), 2)
      ar, EW, pblocking = run(ro, N, k, R, L, S)
      print("Ek2_over_Ek= {}, EW= {}".format(Ek2_over_Ek, EW) )
      
      Ek2_over_Ek_l.append(Ek2_over_Ek)
      EW_l.append(EW)
    blog(Ek2_over_Ek_l=Ek2_over_Ek_l, EW_l=EW_l)
    
    print("ratio = EW/(ER2/ER)")
    for i, EW in enumerate(EW_l):
      Ek2_over_Ek = Ek2_over_Ek_l[i]
      ratio = EW/Ek2_over_Ek
      print("Ek2_over_Ek= {}, ratio= {}".format(Ek2_over_Ek, ratio) )
    log(INFO, "done.")
  
  def check_EW_scaling_wrt_model(N, k, R, L, S):
    log(INFO, "", N=N, k=k, R=R, L=L, S=S)
    sinfo_m['njob'] = 2000*10
    
    ET = L.mean()*sum([X_n_k(S, i, i).mean()*k.pdf(i) for i in k.v_l] )
    ET2 = L.moment(2)*sum([X_n_k(S, i, i).moment(2)*k.pdf(i) for i in k.v_l] )
    EL, EL2 = L.mean(), L.moment(2)
    blog(ET=ET, ET2=ET2, EL=EL, EL2=EL2)
    
    C_moment = lambda i: k.moment(i)*R.moment(i)*L.moment(i)*S.moment(i)
    print(">> C_moment(1)= {}, C_moment(2)= {}".format(C_moment(1), C_moment(2) ) )
    
    def Pr_blocking(ar, ro):
      # narr_atleast_forblocking = (1-ro)*N_times_Cap/(k.moment(1)*R.moment(1) ) - 1
      # blog(narr_atleast_forblocking=narr_atleast_forblocking)
      # ar_ = ar*L.tail(ET)*ET # *L.u_l/10
      # return max(0, \
      #   1 - math.exp(-ar_)*sum([ar_**i/math.factorial(i) for i in range(int(narr_atleast_forblocking) ) ] ) )
      
      alpha = 0.9 # 1/2 # L.cdf(L.u_l/10) # L.cdf(10*EL) # 1/2 # L.cdf(EL)
      # print("alpha= {}".format(alpha) )
      long_jlifetime = EL + math.sqrt((EL2 - EL**2)*alpha/(1-alpha) ) # ET + math.sqrt((ET2 - ET**2)*alpha/(1-alpha) )
      ro_short = ar*L.cdf(long_jlifetime)*C_moment(1)/N_times_Cap
      narr_atleast_forblocking = (1-ro_short)*N_times_Cap / (k.moment(1)*R.moment(1) ) - 1
      blog(narr_atleast_forblocking=narr_atleast_forblocking)
      ar_long = ar*L.tail(long_jlifetime)*long_jlifetime
      return max(0, \
        1 - math.exp(-ar_long)*sum([ar_long**i/math.factorial(i) for i in range(int(narr_atleast_forblocking) ) ] ) )
    
    def EW_givenqed_model(ro):
      return ro/(1-ro) * C_moment(2)/C_moment(1)
    
    def EW_model(ar, ro, pblocking=None):
      if pblocking is None:
        pblocking = Pr_blocking(ar, ro)
      print("pblocking= {}".format(pblocking) )
      return ro/(1-ro) * C_moment(2)/C_moment(1) / 2 * pblocking
    
    EW_l, sim_EW_l = [], []
    # for ro in np.linspace(0.1, 0.9, 9):
    for ro in np.linspace(0.7, 0.9, 3):
      ro = round(ro, 2)
      m = run(ro, N, k, R, L, S)
      ar, sim_EW, sim_pblocking = m['ar'], m['EW'], m['pblocking']
      print("ar= {}, ro= {}".format(ar, ro) )
      
      pblocking = Pr_blocking(ar, ro)
      print("sim_pblocking= {}, pblocking= {}".format(sim_pblocking, pblocking) )
      EW = EW_model(ar, ro, pblocking)
      print("sim_EW= {}, EW= {}".format(sim_EW, EW) )
      sim_EW_l.append(sim_EW)
      EW_l.append(EW)
      
      sim_EW_givenqed = m['EW_givenqed']
      EW_givenqed = EW_givenqed_model(ro)
      print("sim_EW_givenqed= {}, EW_givenqed= {}".format(sim_EW_givenqed, EW_givenqed) )
    blog(EW_l=EW_l, sim_EW_l=sim_EW_l)
    
    # print("ratio = sim_EW/model")
    # for i, sim_EW in enumerate(sim_EW_l):
    #   EW = EW_l[i]
    #   ratio = sim_EW/EW
    #   print("EW= {}, ratio= {}".format(EW, ratio) )
    log(INFO, "done.")
  
  def check_EW_scaling_w_increasing_r(N, k, R, L, S, ro):
    log(INFO, "", N=N, k=k, R=R, L=L, S=S, ro=ro)
    
    # for r in np.linspace(1, 2, 3):
    for r in range(1, 4):
      m = run(ro, N, k, R, L, S, r)
      ar, sim_EW, sim_pblocking = m['ar'], m['EW'], m['pblocking']
      print("ar= {}, ro= {}".format(ar, ro) )
  
  # test(ro=0.4)
  # test(ro=0.65)
  # test(ro=0.9)
  
  # R = Uniform(0.25, 0.75) # Uniform(0.5, 1.5)
  # test(0.9, R)
  
  # R = Uniform(0.25, 0.75) # Uniform(1, 1) # Uniform(0.05, 0.15) # Uniform(0.5, 1.5)
  # check_EW_scaling_wrt_ro(5, R)
  
  # R = Uniform(1.5, 2.5) # Uniform(2, 2)
  # check_EW_scaling_wrt_EL2_over_EL(N, R, ro=0.85)
  
  # L = Exp(0.1, 1)
  # check_EW_scaling_wrt_ER2_over_ER(N, L, ro=0.85)
  
  # R = Uniform(1, 1) # Uniform(1, 1)
  # L = Exp(0.1, 1) # Uniform(1, 1)
  # check_EW_scaling_wrt_Ek2_over_Ek(N, R, L, ro=0.85)
  
  k = BZipf(1, 10) # DUniform(1, 1) # DUniform(1, 4)
  R = Uniform(1, 1)
  L = TPareto(10, 10**6, 4) # Exp(0.1, 1) # Uniform(1, 1)
  S = TPareto(1, 10, 2) # Uniform(1, 1)
  check_EW_scaling_wrt_model(N, k, R, L, S)
  
  log(INFO, "done.")

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
    
    # ED_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
    # EkD_given_kD_g_d = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
    EkD_given_kD_g_d = (Ek*ED - scipy.integrate.quad(lambda x: x*Pr_kD(x), 0, d)[0] ) \
                      / (1 - Pr_kD_leq_d) if Pr_kD_leq_d != 0 else Ek*ED
    
    log(INFO, "", diff=(Ek*ED - (E_kD_given_kD_leq_d*Pr_kD_leq_d + EkD_given_kD_g_d*(1 - Pr_kD_leq_d) ) ) )
    blog(E_kD_given_kD_leq_d=E_kD_given_kD_leq_d, EkD_given_kD_g_d=EkD_given_kD_g_d, Pr_kD_leq_d=Pr_kD_leq_d)
    
    # Using law of total expectation
    ED_given_D_leq_doverk = lambda k: mean(D, given_X_leq_x=True, x=d/k)
    EkD_given_kD_leq_d_ = sum([i*ED_given_D_leq_doverk(i)*k.pdf(i) for i in k.v_l] )
    ED_given_D_g_doverk = lambda k: mean(D, given_X_leq_x=False, x=d/k)
    EkD_given_kD_g_d_ = sum([i*ED_given_D_g_doverk(i)*k.pdf(i) for i in k.v_l] )
    blog(EkD_given_kD_leq_d_=EkD_given_kD_leq_d_, EkD_given_kD_g_d_=EkD_given_kD_g_d_)
  
    sim_E_kD_given_kD_leq_d = gen_sim_E_kD_given_kD_leq_d(d)
    blog(sim_E_kD_given_kD_leq_d=sim_E_kD_given_kD_leq_d)
  
  l, u = a*b, 1000
  for d in np.logspace(math.log10(l), math.log10(u), 10):
    print("\n>> d= {}".format(d) )
    compute(d)
  
  log(INFO, "done.")