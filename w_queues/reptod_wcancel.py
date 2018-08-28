import numpy as np

from patch import *
from rvs import *
from howtorep_exp import *

# #############################################  Model  ########################################## #
ET_MAX = 10000

def ar_ub_reptod_wcancel(ns, J, S):
  EJ, ES = J.mean(), S.mean()
  return ns/EJ/ES

"""
def ET_reptotwo_wcancel(ns, ar, J, S): # B ~ J*S if Ts > X else J*(1-ro)*S2:1
  ar = ar/ns
  EJ, EJ2 = J.moment(1), J.moment(2)
  ES, ES2 = S.moment(1), S.moment(2)
  print("EJ= {}, ES= {}".format(EJ, ES) )
  EV, EV2 = EJ*ES, EJ2*ES2
  print("EV= {}, EV2= {}".format(EV, EV2) )
  
  S21 = X_n_k(S, 2, 1)
  ES21, ES21_2 = moment_ith(1, S21), moment_ith(2, S21)
  
  EV21, EV21_2 = EJ*ES21, EJ2*ES21_2
  print("EV21= {}, EV21_2= {}".format(EV21, EV21_2) )
  # # eq = lambda ro: ro - ar*(ro*EV + (1 - ro)*(1 - ro)*EV21)
  # eq = lambda ro: ro**2*ar*EV21 + ro*(ar*EV - 2*ar*EV21 - 1) + ar*EV21
  # ro = scipy.optimize.brentq(eq, 0.0001, 1)
  # alog("ar= {}, ro= {}".format(ar, ro) )
  # EB = ro*EV + (1 - ro)*(1 - ro)*EV21
  # EB2 = ro*EV2 + (1 - ro)*(1 - ro)**2*EV21_2
  
  # eq = lambda ro: ro - ar*(ro*EV + (1 - ro)*3/4*EV21)
  # Lap_J_ar = laplace(J, ar)
  Pr_T_lt_X = lambda ro: 1 - ro # (1 - ro)/Lap_J_ar
  p = lambda ro: ro*(1 - ro**2) + (1 - ro)*(1-ro) # (1 - ro**2)/2 # (1 - ro)/Lap_J_ar # 1 + (1 - ro)/Lap_J_ar/2 # 1 - ro/4
  
  eq = lambda ro: ro - ar*(ro*EV + Pr_T_lt_X(ro)*p(ro)*EV21)
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ar= {}, \nro= {}".format(ar, ro) )
  ar *= 1 - Pr_T_lt_X(ro)*(1 - p(ro) )
  EB = ro*EV + Pr_T_lt_X(ro)*p(ro)*EV21
  EB2 = ro*EV2 + Pr_T_lt_X(ro)*p(ro)*EV21_2
  
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None
"""
def ET_reptotwo_wcancel(ns, ar, J, S): # B ~ (1-ro)*J*S2:1 if Ts < Tp, Ts > X else J*S
  ar = ar/ns
  EJ, EJ2 = J.moment(1), J.moment(2)
  ES, ES2 = S.moment(1), S.moment(2)
  EV, EV2 = EJ*ES, EJ2*ES2
  
  Pr_rep_makes_diff = lambda ro: (1 - ro**2)/2
  eq = lambda ro: ro - ar*(Pr_rep_makes_diff(ro)*(1-ro)*EV + (1-Pr_rep_makes_diff(ro))*EV)
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ar= {}, \nro= {}".format(ar, ro) )
  EB = Pr_rep_makes_diff(ro)*(1-ro)*EV + (1-Pr_rep_makes_diff(ro))*EV
  EB2 = Pr_rep_makes_diff(ro)*(1-ro)**2*EV2 + (1-Pr_rep_makes_diff(ro))*EV2
  
  ar *= (1-Pr_rep_makes_diff(ro) )
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None

def ET_ED_reptotwo_wcancel_hyperexpJ(ns, ar, J, S):
  ar = ar/ns
  EJ, EJ2 = J.moment(1), J.moment(2)
  ES, ES2 = S.moment(1), S.moment(2)
  EV, EV2 = EJ*ES, EJ2*ES2
  
  Pr_rep_makes_diff = lambda ro: (1 - ro**2)/2
  eq = lambda ro: ro - ar*(Pr_rep_makes_diff(ro)*(1-ro)*EV + (1-Pr_rep_makes_diff(ro))*EV)
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ar= {}, \nro= {}".format(ar, ro) )
  EB = Pr_rep_makes_diff(ro)*(1-ro)*EV + (1-Pr_rep_makes_diff(ro))*EV
  EB2 = Pr_rep_makes_diff(ro)*(1-ro)*EV2 + (1-Pr_rep_makes_diff(ro))*EV2
  # alog("EB= {}, EB2= {}".format(EB, EB2) )
  
  ar *= (1-Pr_rep_makes_diff(ro) )
  ET = EB + ar*EB2/2/(1 - ar*EB)
  alog("ET= {}".format(ET) )
  # return ET if ET < ET_MAX else None
  
  # Y = HyperExp(J.p_l, [mu/(1-ro) for mu in J.mu_l] )
  B_laplace = lambda s: (Pr_rep_makes_diff(ro)*(1-ro) + (1 - Pr_rep_makes_diff(ro) ) )*J.laplace(s)
  B_pdf = lambda t: mpmath.invertlaplace(B_laplace, t, method='talbot')
  B_moment = lambda i: scipy.integrate.quad(lambda t: t**i * B_pdf(t), 0, np.inf)
  
  # EB_, EB2_ = B_moment(1), B_moment(2)
  # alog("EB_= {}, EB2_= {}".format(EB_, EB2_) )
  
  ro = ar*EB
  def T_laplace(s):
    Bs = B_laplace(s)
    return (1 - ro)*s/(s - ar + ar*Bs)*Bs
  T_pdf = lambda t: mpmath.invertlaplace(T_laplace, t, method='talbot')
  T_moment = lambda i: scipy.integrate.quad(lambda t: t**i * T_pdf(t), 0, 500) # mpmath.quad(lambda t: t**i * T_pdf(t), [0, mpmath.inf] )
  # ET_ = T_moment(1)
  # alog("ET_= {}".format(ET_) )
  
  # for t in np.linspace(0.01, 20, 100):
  #   print("T_pdf({})= {}".format(t, T_pdf(t) ) )

def ETlb_reptotwo_wcancel(ns, ar, J, S): # B ~ J*S if Ts > X else J
  ar = ar/ns
  EJ, EJ2 = J.moment(1), J.moment(2)
  ES, ES2 = S.moment(1), S.moment(2)
  
  ro = ar*EJ/(1 - ar*EJ*ES + ar*EJ)
  alog("ar= {}, ro= {}".format(ar, ro) )
  EB = ro*EJ*ES + (1 - ro)*EJ
  EB2 = ro*EJ**2*ES**2 + (1 - ro)*EJ**2
  
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None

# ##################################  Reptod-ifidle-wcancel  ##################################### #
def ET_reptod_ifidle(ns, d, J, S, ar):
  def EB1_mth(m):
    return J.moment(m)*S.moment(m)
  
  ar_toidleq = ar/ns*d
  def Pr_Si1_g_s(i, s):
    return S.tail(s)**i
  def ESi1_mth(i, m):
    return scipy.integrate.quad(lambda s: m*s**(m-1) * Pr_Si1_g_s(i, s), 0, np.inf)[0]
  def EBi1_mth(i, m):
    return J.moment(m)*ESi1_mth(i, m)
  
  Pr_jobfindsidle = lambda ro: 1 - ro**d
  def ET_given_jobfindsidle(ro):
    return sum([EBi1_mth(i, 1) * (1-ro)**i * ro**(d-i) * binom(d, i) / (1 - ro**d) for i in range(1, d+1) ] )
  
  ar_mg1efs = lambda ro: ar/ns *ro**(d-1)
  def ESe_mth(ro, m):
    return ro**(d-1) * S.moment(m) \
           + sum([ESi1_mth(i, m) * binom(d-1, i-1)*(1-ro)**(i-1)*ro**(d-i) for i in range(2, d+1) ] )
  def EBe_mth(ro, m):
    return J.moment(m)*ESe_mth(ro, m)
  
  def ET_given_jobfindsnoidle(ro):
    ar_ = ar_mg1efs(ro)
    EB1, EB12 = EB1_mth(1), EB1_mth(2)
    EBe, EBe2 = EBe_mth(ro, 1), EBe_mth(ro, 2)
    return EB1 + ar_*EB12/2/(1 - ar_*EB1) + ar_*(EBe2 - EB12)/2/(1 - ar_*(EB1 - EBe) )
  
  Elengthofbusyperiod = lambda ro: EBe_mth(ro, 1)/(1 - ar_mg1efs(ro)*EB1_mth(1) )
  eq = lambda ro: ro - Elengthofbusyperiod(ro)/(Elengthofbusyperiod(ro) + 1/ar_toidleq)
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ro= {}".format(ro) )
  alog("naive ro= {}".format(ar/ns*EB1_mth(1) ) )
  
  ET = Pr_jobfindsidle(ro)*ET_given_jobfindsidle(ro) + \
       (1 - Pr_jobfindsidle(ro))*ET_given_jobfindsnoidle(ro)
  alog("ET= {}".format(ET) )

def plot_reptod_ifidle():
  ns, d = 100, 2
  J = HyperExp([0.9, 0.1], [1, 0.01] ) # TPareto(1, 10**4, 1.1) # Exp(1) # DUniform(1, 1)
  S = Bern(1, 10, 0.1) # Exp(1) # Pareto(1, 2)
  T = ns*4000 # 10000
  alog("ns= {}, d= {}, J= {}, S= {}, T= {}".format(ns, d, J, S, T) )
  
  EJ, ES = J.mean(), S.mean()
  EB = EJ*ES
  print("EJ= {}, ES= {}, EB= {}".format(EJ, ES, EB) )
  ar_ub = 0.99*ns/EB
  nf = 1
  for ar in np.linspace(0.01, ar_ub, 7):
    print("> ar= {}".format(ar) )
    
    sching_m = {'name': 'reptod-ifidle', 'd': d, 's_len': d}
    rosim, ETsim, EDsim = sim_reptod(nf, ns, sching_m, J, S, ar, T, jg_type='poisson')
    print("sching_m= {}".format(sching_m) )
    print("rosim= {}, ETsim= {}, EDsim= {}".format(rosim, ETsim, EDsim) )
    ET_reptod_ifidle(ns, d, J, S, ar)
    print("\n")
    
    # sching_m = {'name': 'reptod-ifidle-wcancel', 'd': d, 's_len': d, 'L': 0}
    # rosim, ETsim, EDsim = sim_reptod(nf, ns, sching_m, J, S, ar, T, jg_type='poisson')
    # print("sching_m= {}".format(sching_m) )
    # print("rosim= {}, ETsim= {}, EDsim= {}".format(rosim, ETsim, EDsim) )
    # ET_reptod_ifle_wcancel(ns, d, J, S, ar)
    # print("\n\n")

# ###############################  Reptod-ifidle-wcancel  ############################### #
def ET_reptod_ifle_wcancel(ns, d, J, S, ar):
  def EB1_mth(m):
    return J.moment(m)*S.moment(m)
  
  ar_toidleq = lambda ro: sum([ar/ns*d/(1+i) * binom(d-1, i)*(1-ro)**i*ro**(d-1-i) for i in range(d) ] )
  def Pr_Sred_g_s(ro, s):
    return 1 - scipy.integrate.quad(lambda v: S.pdf(v)*math.exp(-ar_toidleq(ro)*v), 0, s)[0]
  def Pr_Si1_g_s(ro, i, s):
    return S.tail(s) * Pr_Sred_g_s(ro, s)**(i-1)
  def ESi1_mth(ro, i, m):
    return scipy.integrate.quad(lambda s: m*s**(m-1) * Pr_Si1_g_s(ro, i, s), 0, np.inf)[0]
  def EBi1_mth(ro, i, m):
    return J.moment(m)*ESi1_mth(ro, i, m)
  
  Pr_jobfindsidle = lambda ro: 1 - ro**d
  def ET_given_jobfindsidle(ro):
    return sum([EBi1_mth(ro, i, 1) * (1-ro)**i * ro**(d-i) * binom(d, i) / (1 - ro**d) for i in range(1, d+1) ] )
  
  ar_mg1efs = lambda ro: ar/ns *ro**(d-1)
  def ESe_mth(ro, m):
    return ro**(d-1) * S.moment(m) \
           + sum([ESi1_mth(ro, i, m) * binom(d-1, i-1)*(1-ro)**(i-1)*ro**(d-i) for i in range(2, d+1) ] )
  def EBe_mth(ro, m):
    return J.moment(m)*ESe_mth(ro, m)
  
  def ET_given_jobfindsnoidle(ro):
    ar_ = ar_mg1efs(ro)
    EB1, EB12 = EB1_mth(1), EB1_mth(2)
    EBe, EBe2 = EBe_mth(ro, 1), EBe_mth(ro, 2)
    return EB1 + ar_*EB12/2/(1 - ar_*EB1) + ar_*(EBe2 - EB12)/2/(1 - ar_*(EB1 - EBe) )
  
  Elengthofbusyperiod = lambda ro: EBe_mth(ro, 1)/(1 - ar_mg1efs(ro)*EB1_mth(1) )
  eq = lambda ro: ro - Elengthofbusyperiod(ro)/(Elengthofbusyperiod(ro) + 1/ar_toidleq(ro) )
  ro = scipy.optimize.brentq(eq, 0.0001, 1)
  alog("ro= {}".format(ro) )
  alog("naive ro= {}".format(ar/ns*EB1_mth(1) ) )
  
  ET = Pr_jobfindsidle(ro)*ET_given_jobfindsidle(ro) + \
       (1 - Pr_jobfindsidle(ro))*ET_given_jobfindsnoidle(ro)
  alog("ET= {}".format(ET) )

def plot_reptod_ifidle_wcancel():
  ns, d = 10, 2
  sching_m = {'name': 'reptod-ifidle-wcancel', 'd': d, 's_len': d, 'L': 0}
  J = DUniform(1, 1)
  S = Exp(0.1)
  T = ns*5000 # 10000
  alog("ns= {}, d= {}, J= {}, S= {}, T= {}".format(ns, d, J, S, T) )
  
  EB = J.mean()*S.mean()
  ar_ub = 0.9*ns/EB
  
  nf = 1
  for ar in np.linspace(0.01, ar_ub, 5):
    print("> ar= {}".format(ar) )
    rosim, ETsim, EDsim = sim_reptod(nf, ns, sching_m, J, S, ar, T, jg_type='poisson')
    print("sching_m= {}".format(sching_m) )
    print("rosim= {}, ETsim= {}, EDsim= {}".format(rosim, ETsim, EDsim) )
    
    # sching_m = {'name': 'reptod', 'd': d, 's_len': d}
    # sching_m = {'name': 'reptod', 'd': 1, 's_len': 1}
    # rosim, ETsim, EDsim = sim_reptod(nf, ns, sching_m, J, S, ar, T, jg_type='poisson')
    # print("sching_m= {}".format(sching_m) )
    # print("rosim= {}, ETsim= {}, EDsim= {}".format(rosim, ETsim, EDsim) )
    
    ro, ET = ET_reptod_ifle_wcancel(ns, d, J, S, ar)
    print("ro= {}, ET= {}".format(ro, ET) )
    print("\n\n")
  

# #############################################  Sim  ############################################ #
def sim_reptod(nf, ns, sching_m, J, S, ar, T, jg_type='poisson'):
  ro, ET, ET2, ED = 0, 0, 0, 0
  for _ in range(nf):
    env = simpy.Environment()
    jg = JG(env, ar, DUniform(1, 1), J, T, jg_type)
    mq = MultiQ_wRep(env, ns, T, sching_m, S)
    jg.out = mq
    jg.init()
    env.run()
    
    if 'd' in sching_m:
      r_numj_l = list(range(sching_m['d'] ) )
      for jid, r in mq.jq.jid_r_m.items():
        r_numj_l[r] += 1
      r_numj_l[0] = T - sum(r_numj_l)
      r_freqj_l = [nj/T for nj in r_numj_l]
      print("r_freqj_l= {}".format(r_freqj_l) )
    print("avg load across servers= {}".format([mq.q_l[i].busy_t/env.now for i in range(ns) ] ) )
    EBsim = np.mean([np.mean(mq.q_l[i].EB_l) for i in range(ns) ] )
    print("EBsim= {}".format(EBsim) )
    
    ro += np.mean([mq.q_l[i].busy_t/env.now for i in range(ns) ] )
    ET += np.mean([np.mean(mq.q_l[i].lt_l) for i in range(ns) ] )
    ED += np.mean([mq.jid_info_m[t+1]['T'] for t in range(T) ] )
    # ET2 += ET**2
  return ro/nf, ET/nf, ED/nf # , ET2/nf

def plot_reptod_wcancel():
  ns = 10
  T = 50000
  J = DUniform(1, 1) # HyperExp([0.8, 0.2], [1, 0.01] ) # TPareto(1, 1000*10, 1.1) # TPareto(1, 10**10, 1.1)
  S = Exp(0.1) # DUniform(1, 1) # Bern(1, 27, 0.2) # Dolly() # TPareto(1, 12, 3)
  
  ar_ub = 0.9*ar_ub_reptod_wcancel(ns, J, S)
  alog("ns= {}, T= {}, J= {}, S= {}, ar_ub= {}".format(ns, T, J, S, ar_ub) )
  
  ar_l = []
  ET_sim_l, ET_l, ETlb_l = [], [], []
  
  sim = True # False
  if not sim:
    pass
  
  nf = 1
  def compare(d):
    for ar in [*np.linspace(0.05, 0.8*ar_ub, 5, endpoint=False), *np.linspace(0.8*ar_ub, ar_ub, 5, endpoint=False) ]:
      print("\n> ar= {}".format(ar) )
      
      # sching_m = {'reptod-wcancel': 0, 'd': 1}
      # ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
      # print("sching_m= {}, \nED_sim= {}".format(sching_m, ED_sim) )
      
      sching_m = {'name':'reptod', 'd': d, 's_len': d}
      ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
      print("sching_m= {}, \nED_sim= {}".format(sching_m, ED_sim) )
      
      sching_m = {'name': 'reptod-ifidle', 'd': d, 's_len': d}
      ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
      print("sching_m= {}, \nED_sim= {}".format(sching_m, ED_sim) )
      
      sching_m = {'name': 'reptod-wcancel', 'd': d, 's_len': d}
      ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
      print("sching_m= {}, \nED_sim= {}".format(sching_m, ED_sim) )
      
      # sching_m = {'reptod-wcancel': 0, 'd': 3}
      # ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
      # print("sching_m= {}, \nED_sim= {}".format(sching_m, ED_sim) )
  
  def plot_(d):
    print("d= {}".format(d) )
    # for ar in np.linspace(0.05, ar_ub + 0.1, 2):
    # for ar in np.linspace(ar_ub, ar_ub, 1):
    # for ar in [*np.linspace(0.05, 0.8*ar_ub, 5, endpoint=False), *np.linspace(0.8*ar_ub, ar_ub, 5, endpoint=False) ]:
    for ar in np.linspace(0.01, ar_ub, 5):
      print("\n> ar= {}".format(ar) )
      ar_l.append(ar)
      
      if sim:
        sching_m = {'reptod-wcancel': 0, 'd': d}
        ro_sim, ET_sim, ED_sim = sim_reptod(nf, ns, sching_m, J, S, T, ar)
        print("sching_m= {}, \nro_sim= {}, \nET_sim= {}".format(sching_m, ro_sim, ET_sim) )
        ET_sim_l.append(ET_sim)
      
      if d == 2:
        ET = ET_reptotwo_wcancel(ns, ar, J, S)
        print("ET= {}".format(ET) )
        ET_l.append(ET)
        
        # ET_ED_reptotwo_wcancel_hyperexpJ(ns, ar, J, S)
        
    # print("ET_sim_l=\n{}".format(pprint.pformat(ET_sim_l) ) )  
    plot.plot(ar_l, ET_sim_l, label=r'Simulation, $d= {}$'.format(d), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
    if d == 2:
      plot.plot(ar_l, ET_l, label=r'Approximation, $d= {}$'.format(d), color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  
  # compare()
  
  plot_(d=2)
  # plot_(d=3)
  
  plot.title(r'Redundancy-d-w/cancel, $n= {}$'.format(ns) + "\n" + r'$J \sim {}$, $S \sim {}$'.format(J, S) )
  plot.xlabel(r'$\lambda$', fontsize=14)
  plot.ylabel(r'Average response time', fontsize=14)
  plot.legend()
  plot.savefig("ET_reptod_wcancel.pdf")
  plot.gcf().clear()
  log(WARNING, "done.")

if __name__ == "__main__":
  # plot_reptod_wcancel()
  plot_reptod_ifidle()
  # plot_reptod_ifidle_wcancel()
