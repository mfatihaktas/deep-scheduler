from modeling import *

# def ar_for_ro(ro, N, Cap, k, D, S):
#   return ro*N*Cap/k.mean()/D.mean()/S.mean()

def ar_for_ro(ro, N, Cap, k, R, L, S):
  return ro*N*Cap/k.mean()/R.mean()/L.mean()/S.mean()

def EW_MMc(ar, EX, c):
  ro = ar*EX/c
  C = 1/(1 + (1-ro)*G(c+1)/(c*ro)**c * sum([(c*ro)**k/G(k+1) for k in range(c) ] ) )
  # EN = ro/(1-ro)*C + c*ro
  return C/(c/EX - ar)

def EW_MGc(ar, X, c):
  EX2, EX = X.moment(2), X.moment(1)
  CoeffVar = math.sqrt(EX2 - EX**2)/EX
  return (1 + CoeffVar**2)/2 * EW_MMc(ar, EX, c)

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
    'ar': ar, 'njob': 2000*10, 'nworker': N, 'wcap': Cap,
    'lifetime_rv': Pareto(b, beta),
    'reqed_rv': DUniform(1, 1),
    'k_rv': k,
    'straggle_m': {'slowdown': lambda load: S.sample() } }
  mapping_m = {'type': 'spreading'}
  sching_m = {'type': 'expand_if_totaldemand_leq', 'r': r, 'threshold': None}
  # blog(sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  check_MGc_assumption()
