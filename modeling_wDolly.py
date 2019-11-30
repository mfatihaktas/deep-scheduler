import numpy as np

from rvs import Dolly
from math_utils import *
from plot_utils import *

def Pr_Xnk_leq_x(X, n, k, x):
  # log(INFO, "x= {}".format(x) )
  cdf = 0
  for i in range(k, n+1):
    cdf += binom_(n, i) * X.cdf(x)**i * X.tail(x)**(n-i)
  return cdf

def EXnk(X, n, k, m=1):
  if k == 0:
    return 0
  
  if m == 1:
    # EXnk, abserr = scipy.integrate.quad(lambda x: 1 - Pr_Xnk_leq_x(X, n, k, x), 0.0001, np.Inf) # 2*X.u_l
    EXnk = float(mpmath.quad(lambda x: 1 - Pr_Xnk_leq_x(X, n, k, x), [0.0001, 10*X.u_l] ) )
  else:
    # EXnk, abserr = scipy.integrate.quad(lambda x: m*x**(m-1) * (1 - Pr_Xnk_leq_x(X, n, k, x)), 0.0001, np.Inf)
    EXnk = float(mpmath.quad(lambda x: m*x**(m-1) * (1 - Pr_Xnk_leq_x(X, n, k, x) ), [0.0001, 10*X.u_l] ) )
  return EXnk

def ECnk(X, n, k):
  if k == 0:
    return 0
  
  EC = 0
  for i in range(1, k):
    EC += EXnk(X, n, i)
  EC += (n-k+1)*EXnk(X, n, k)
  return EC

def plot_cdf_X(X):
  x_l, Pr_X_leq_x_l = [], []
  for x in np.linspace(0, 30, 100):
    x_l.append(x)
    Pr_X_leq_x_l.append(X.cdf(x) )
  plot.plot(x_l, Pr_X_leq_x_l, c='blue', marker='x', ls=':', mew=0.1, ms=8)
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=14, numpoints=1)
  plot.xlabel(r'$x$', fontsize=fontsize)
  plot.ylabel(r'$\Pr\{X \leq x\}$', fontsize=fontsize)
  
  plot.title(r'$X \sim {}$'.format(X.to_latex() ), fontsize=fontsize)
  fig = plot.gcf()
  fig.set_size_inches(4, 4)
  plot.savefig('plot_cdf_X.png', bbox_inches='tight')
  fig.clear()
  log(INFO, "done.")

def redsmall_ES_wSl(k, r, D, Sl, d=None, red='coding'):
  if d is None:
    return D.mean()*sum([EXnk(Sl, i, i)*k.pdf(i) for i in k.v_l] )
  
  ED_given_D_leq_doverk = lambda k: D.mean_given_leq_x(d/k)
  return redsmall_ES_wSl(k, r, D, Sl, d=None, red=red) \
          + sum([(EXnk(Sl, i*r, i) - EXnk(Sl, i, i) )*ED_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )
          # + sum([(ES_k_n_pareto(i, i*r, a, alpha) - ES_k_n_pareto(i, i, a, alpha) )*ED_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def redsmall_ES2_wSl(k, r, D, Sl, d=None, red='coding'):
  if d is None:
    return D.moment(2)*sum([EXnk(Sl, i, i, m=2)*k.pdf(i) for i in k.v_l] )
  
  ED2_given_D_leq_doverk = lambda k: moment(D, 2, given_X_leq_x=True, x=d/k)
  return redsmall_ES2_wSl(k, r, D, Sl, d=None, red=red) \
          + sum([(EXnk(Sl, i*r, i, m=2) - EXnk(Sl, i, i, m=2) )*ED2_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def redsmall_EC_wSl(k, r, D, Sl, d=None, red='coding'):
  if d is None:
    return k.mean()*D.mean()*Sl.mean()
  
  ED_given_D_leq_doverk = lambda k: D.mean_given_leq_x(d/k)
  return redsmall_EC_wSl(k, r, D, Sl, d=None, red=red) \
         + sum([(ECnk(Sl, i*r, i) - i*Sl.mean())*ED_given_D_leq_doverk(i)*D.cdf(d/i)*k.pdf(i) for i in k.v_l] )

def ar_for_ro0(ro0, N, Cap, k, r, D, Sl):
  return ro0*N*Cap/k.mean()/D.mean()/Sl.mean()

def redsmall_ET_EW_Prqing_wMGc_wSl(ro0, N, Cap, k, r, D, Sl, d, red='coding'):
  '''Using the result for M/M/c to approximate E[T] in M/G/c.
     [https://en.wikipedia.org/wiki/M/G/k_queue]
  '''
  ar = ar_for_ro0(ro0, N, Cap, k, r, D, Sl)
  
  ES = redsmall_ES_wSl(k, r, D, Sl, d, red)
  ES2 = redsmall_ES2_wSl(k, r, D, Sl, d, red)
  EC = redsmall_EC_wSl(k, r, D, Sl, d, red)
  log(INFO, "d= {}".format(d), ES=ES, ES2=ES2, EC=EC)
  
  EW, Prqing = MGc_EW_Prqing(ar, N*Cap*ES/EC, ES, ES2)
  if EW < 0:
    # log(ERROR, "!!!", EW=EW, Prqing=Prqing, ES=ES, ES2=ES2, EC=EC)
    # return None, None, None
    # return (ES + abs(EW))**2, None, None
    return 10**6, None, None
  
  ET = ES + EW
  # log(INFO, "d= {}, ro= {}, ES= {}, EW= {}, ET= {}".format(d, ro, ES, EW, ET) )
  # log(INFO, "d= {}, ro= {}".format(d, ro) )
  # return round(ET, 2), round(EW, 2), round(Prqing, 2)
  return ET, EW, Prqing

def redsmall_approx_ET_EW_Prqing_wMGc_wSl(ro0, N, Cap, k, r, D, Sl, d, red='coding'):
  ar = ar_for_ro0(ro0, N, Cap, k, r, D, Sl)
  ro = ro0
  
  ES = redsmall_ES_wSl(k, r, D, Sl, d, red)
  # ES2 = redsmall_ES2_wSl(k, r, D, Sl, d, red)
  # EC = redsmall_EC_wSl(k, r, D, Sl, d, red)
  log(INFO, "d= {}".format(d), ar=ar, ES=ES) # , ES2=ES2, EC=EC
  
  EW = 1/ar * ro**2/(1 - ro)
  
  ET = ES + EW
  return ET, EW, ro

def plot_ET(N, Cap, k, r, D, Sl, red='coding'):
  def plot_(ro0):
    log(INFO, "ro0= {}".format(ro0) )
    d_l, ET_l = [], []
    for d in np.linspace(D.l_l, D.mean()*15, 7):
      ET, EW, Prqing = redsmall_ET_EW_Prqing_wMGc_wSl(ro0, N, Cap, k, r, D, Sl, d, red='coding') # redsmall_ES_wSl(k, r, D, Sl, d, red)
      log(INFO, "d= {}, ET= {}, EW= {}, Prqing= {}".format(d, ET, EW, Prqing) )
      
      if ET > 150:
        break
      d_l.append(d)
      ET_l.append(ET)
    plot.plot(d_l, ET_l, label=r'$\rho_0= {}$'.format(ro0), c=next(darkcolor_c), marker=next(marker_c), ls=':', mew=0.1, ms=8)
  
  plot_(ro0=0.8)
  # plot_(ro0=0.9)
  
  fontsize = 20
  plot.legend(loc='best', framealpha=0.5, fontsize=14, numpoints=1)
  plot.xlabel(r'$d$', fontsize=fontsize)
  plot.ylabel(r'$E[T]$', fontsize=fontsize)
  
  plot.title(r'$r= {}$, $k \sim {}$'.format(r, k.to_latex() ) + "\n" \
             + r'$D \sim {}$, $Sl \sim {}$'.format(D.to_latex(), Sl.to_latex() ), fontsize=fontsize)
  fig = plot.gcf()
  fig.set_size_inches(4, 4)
  plot.savefig('plot_ET.png', bbox_inches='tight')
  fig.clear()
  log(INFO, "done.")

if __name__ == "__main__":
  X = Dolly()
  print("EX= {}".format(X.mean() ) )
  
  def EXnk_(n, k):
    EX_ = EXnk(X, n, k)
    print("n= {}, k= {}, EXnk= {}".format(n, k, EX_) )
  
  # EXnk_(n=10, k=10)
  # EXnk_(n=10, k=8)
  # EXnk_(n=10, k=5)
  
  N, Cap = 20, 10
  k = BZipf(1, 10)
  r = 2
  D = Pareto(10, 3)
  Sl = Dolly()
  plot_ET(N, Cap, k, r, D, Sl)
  