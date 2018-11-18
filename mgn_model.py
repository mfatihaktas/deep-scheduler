from modeling import *

def ar_for_ro(ro, N, Cap, k, D, S):
  return ro*N*Cap/k.mean()/D.mean()/S.mean()

def EW_MMc(ar, EX, c):
  ro = ar*EX/c
  C = 1/(1 + (1-ro)*G(c+1)/(c*ro)**c * sum([(c*ro)**k/G(k+1) for k in range(c) ] ) )
  # EN = ro/(1-ro)*C + c*ro
  return C/(c/EX - ar)

def EW_MGc(ar, X, c):
  EX2, EX = X.moment(2), X.moment(1)
  CoeffVar = math.sqrt(EX2 - EX**2)/EX
  return (1 + CoeffVar**2)/2 * EW_MMc(ar, EX, c)

def check_MGN_assumption():
  N, Cap = 10, 1
  r = 1
  D = Exp(1, 1)
  S = DUniform(1, 1)
  sinfo_m.update({
    'njob': 2000*10,
    'totaldemand_rv': D,
    'straggle_m': {'slowdown': lambda load: S.sample() } } )
  sching_m = {'type': 'plain', 'r': r}
  blog(sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  def run(N, Cap, ro):
    print("\n")
    log(INFO, "N= {}, Cap= {}, ro= {}".format(N, Cap, ro) )
    ar = round(ar_for_ro(ro, N, Cap, k, D, S), 2)
    
    sinfo_m.update({'nworker': N, 'wcap': Cap, 'ar': ar} )
    sim_m = sim(sinfo_m, mapping_m, sching_m, "N{}_C{}".format(N, Cap) )
    blog(sim_m=sim_m)
    
    sim_EW = sim_m['waittime_mean']
    EW = EW_MGc(ar, D, N*Cap)
    print("sim_EW= {}, EW= {}".format(sim_EW, EW) )
  
  def test(ro):
    print("---------------")
    run(N=1, Cap=10, ro=ro)
    run(N=2, Cap=5, ro=ro)
    run(N=5, Cap=2, ro=ro)
    run(N=10, Cap=1, ro=ro)
  
  test(ro=0.4)
  test(ro=0.65)
  test(ro=0.9)
  
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
  
  check_MGN_assumption()
