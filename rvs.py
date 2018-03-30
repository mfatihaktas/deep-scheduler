import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot

import math, random, numpy, csv
from cmath import *
from scipy.stats import *
from patch import *

class RV(): # Random Variable
  def __init__(self, l_l, u_l):
    self.l_l = l_l
    self.u_l = u_l

class Exp(RV):
  def __init__(self, mu, D=0):
    RV.__init__(self, l_l=D, u_l=float("inf") )
    self.D = D
    self.mu = mu
  
  def __str__(self):
    return r'Exp(D={}, \mu={})'.format(self.D, self.mu)
  
  def tail(self, x):
    if x <= self.l_l:
      return 1
    return math.exp(-self.mu*(x - self.D) )
  
  def cdf(self, x):
    if x <= self.l_l:
      return 0
    return 1 - math.exp(-self.mu*(x - self.D) )
  
  def pdf(self, x):
    if x <= self.l_l:
      return 0
    return self.mu*math.exp(-self.mu*(x - self.D) )
  
  def mean(self):
    return self.D + 1/self.mu
  
  def var(self):
    return 1/self.mu**2
  
  def moment(self, i):
    return moment_ith(i, self)
  
  def laplace(self, s):
    if self.D > 0:
      log(ERROR, "D= {} != 0".format(D) )
    return self.mu/(s + self.mu)
  
  def gen_sample(self):
    return self.D + random.expovariate(self.mu)

class HyperExp(RV):
  def __init__(self, p_l, mu_l):
    RV.__init__(self, l_l=0, u_l=float("inf") )
    self.p_l = p_l
    self.mu_l = mu_l
    
    self.X_l = [Exp(mu) for mu in mu_l]
    
    self.i_l = [i for i in range(len(self.p_l) ) ]
    self.dist_for_gensample = scipy.stats.rv_discrete(name='hyperexp', values=(self.i_l, self.p_l) )
  
  def __str__(self):
    return r'HyperExp(p= {}, \mu= {})'.format(self.p_l, self.mu_l)
  
  def pdf(self, x):
    return sum([self.p_l[i]*X.pdf(x) for i, X in enumerate(self.X_l) ] )
  
  def cdf(self, x):
    return sum([self.p_l[i]*X.cdf(x) for i, X in enumerate(self.X_l) ] )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return sum([self.p_l[i]*X.mean() for i, X in enumerate(self.X_l) ] )
  
  def moment(self, m):
    return sum([self.p_l[i]*X.moment(m) for i, X in enumerate(self.X_l) ] )
  
  def laplace(self, s):
    return sum([self.p_l[i]*X.laplace(s) for i, X in enumerate(self.X_l) ] )
  
  def gen_sample(self):
    i = self.dist_for_gensample.rvs()
    return self.X_l[i].gen_sample()

class Pareto(RV):
  def __init__(self, loc, a):
    RV.__init__(self, l_l=loc, u_l=float("inf") )
    self.loc = loc
    self.a = a
  
  def __str__(self):
    return "Pareto(loc= {}, a= {})".format(self.loc, self.a)
  
  def tolatex(self):
    return r'Pareto(\min= {}, \alpha= {})'.format(self.loc, self.a)
  
  def tail(self, x):
    if x <= self.l_l:
      return 1
    return (self.loc/x)**self.a
  
  def cdf(self, x):
    if x <= self.l_l:
      return 0
    return 1 - (self.loc/x)**self.a
  
  def pdf(self, x):
    if x <= self.l_l:
      return 0
    return self.a*self.loc**self.a / x**(self.a+1)
  
  def dpdf_dx(self, x):
    if x <= self.l_l:
      return 0
    return sympy.mpmath.diff(lambda y: self.a*self.loc**self.a / y**(self.a+1), x)
  
  def mean(self):
    if self.a <= 1:
      log(WARNING, "Mean is Infinity; a= {} <= 1".format(self.a) )
      return float("inf")
    else:
      return self.loc*self.a/(self.a-1)
  
  def var(self):
    if self.a <= 2:
      log(WARNING, "Variance is Infinity; a= {} <= 2".format(self.a) )
      return float("inf")
    else:
      return self.a*self.loc**2 / (self.a-1)**2/(self.a-2)
  
  def gen_sample(self):
    return ((numpy.random.pareto(self.a, 1) + 1)*self.loc)[0]
    # return pareto.ppf(numpy.random.uniform(0, 1), b=self.a, scale=self.loc)

class TPareto(): # Truncated
  def __init__(self, l, u, a):
    RV.__init__(self, l_l=l, u_l=u)
    self.l = l
    self.u = u
    self.a = a
  
  def __str__(self):
    return "TPareto(l= {}, u= {}, a= {})".format(self.l, self.u, self.a)
  
  def tolatex(self):
    return r'TPareto(\min= {}, \max= {}, \alpha= {})'.format(self.l, self.u, self.a)
  
  def pdf(self, x):
    if x < self.l: return 0
    elif x >= self.u: return 0
    else:
      return self.a*self.l**self.a * 1/x**(self.a+1) / (1 - (self.l/self.u)**self.a)
  
  def cdf(self, x):
    if x < self.l: return 0
    elif x >= self.u: return 1
    else:
      return (1 - (self.l/x)**self.a)/(1 - (self.l/self.u)**self.a)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.moment(1)
  
  def moment(self, k):
    if k == self.a:
      return math.log(self.u_l/self.l)
    else:
      return self.a*self.l**k/(self.a-k) * \
             (1 - (self.l/self.u)**(self.a-k))/(1 - (self.l/self.u)**self.a)
  
  def gen_sample(self):
    r = random.uniform(0, 1)
    s = self.l*(1 - r*(1-(self.l/self.u)**self.a) )**(-1/self.a)
    if s < self.l or s > self.u:
      log(ERROR, "illegal sample! s= {}".format(s) )
      return None
    return s

def plot_gensample_check():
  l, u, a = 1, 10**5, 2
  rv = TPareto(l, u, a)
  
  x_l = []
  for i in range(10**5):
    x_l.append(rv.gen_sample() )
  x_l = numpy.sort(x_l)
  x_l = x_l[::-1]
  # i_ = None
  # for i in range(len(x_l)-1, 0, -1):
  #   if x_l[i] > 1.01: i_ = i; break
  # x_l = x_l[:i_]
  y_l = numpy.arange(x_l.size)/x_l.size
  plot.plot(x_l, y_l, marker=next(marker), color=next(dark_color), linestyle=':', mew=mew, ms=ms)
  
  y_l = []
  for x in x_l:
    y_l.append(rv.tail(x) )
  plot.plot(x_l, y_l, label=r'$Pareto(l= %.2f, u= %.2f, \alpha= %.2f)$' % (l, u, a), color=next(dark_color), linestyle='-')
  plot.legend()
  plot.xscale('log')
  plot.yscale('log')
  plot.xlabel(r'$x$', fontsize=13)
  plot.ylabel(r'$p(X > x)$', fontsize=13)
  plot.title(r'$X \sim$ {}'.format(rv) )
  plot.savefig("plot_gensample_check.png")
  plot.gcf().clear()

class Google(RV):
  def __init__(self, k):
    RV.__init__(self, l_l=0, u_l=float("inf") )
    
    self.k = k
    self.sample_l = []
    # with open("filtered_task_lifetimes_for_jobs_w_num_task_{}.dat".format(k), mode="rt") as f:
    with open("task_lifetimes_for_jobs_w_num_task_{}.dat".format(k), mode="rt") as f:
      reader = csv.reader(f)
      for line in reader:
        self.sample_l.append(float(line[0] ) )
    self.sample_l.sort()
    self.num_sample = len(self.sample_l)
  
  def __str__(self):
    return "Google(k= ".format(self.k)
  
  def mean(self):
    return sum(self.sample_l)/self.num_sample
  
  def gen_sample(self):
    return self.sample_l[math.floor(self.num_sample*random.random() ) ]

class SimRV(RV):
  def __init__(self, sample_l):
    RV.__init__(self, l_l=min(sample_l), u_l=max(sample_l) )
    
    self.sample_l = sample_l
    self.num_sample = len(self.sample_l)
  
  def __str__(self):
    return "SimRV"
  
  def mean(self):
    return sum(self.sample_l)/self.num_sample
  
  def gen_sample(self):
    return self.sample_l[math.floor(self.num_sample*random.random() ) ]

class Dolly(RV):
  # Kristen et al. A Better Model for Job Redundancy: Decoupling Server Slowdown and Job Size
  def __init__(self):
    RV.__init__(self, l_l=1, u_l=12)
    
    self.v = numpy.arange(1, 13)
    self.p = [0.23, 0.14, 0.09, 0.03, 0.08, 0.1, 0.04, 0.14, 0.12, 0.021, 0.007, 0.002]
    self.dist = scipy.stats.rv_discrete(name='dolly', values=(self.v, self.p) )
  
  def __str__(self):
    return "Dolly[{}, {}]".format(self.l_l, self.u_l)
  
  def tolatex(self):
    return "Dolly"
  
  def pdf(self, x):
    return self.dist.pmf(x) if (x >= self.l_l and x <= self.u_l) else 0
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return float(self.dist.cdf(math.floor(x) ) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    return self.moment(1)
  
  def moment(self, m):
    return sum([self.v[i]**m*self.p[i] for i in range(12) ] )
  
  def gen_sample(self):
    return random.randint(self.l_l, self.u_l)
  
  def gen_sample(self):
    u = random.uniform(0, 1)
    # if u <= 0.23: return 1 + u/100
    # u -= 0.23
    # if u <= 0.14: return 2 + u/100
    # u -= 0.14
    # if u <= 0.09: return 3 + u/100
    # u -= 0.09
    # if u <= 0.03: return 4 + u/100
    # u -= 0.03
    # if u <= 0.08: return 5 + u/100
    # u -= 0.08
    # if u <= 0.1: return 6 + u/100
    # u -= 0.1
    # if u <= 0.04: return 7 + u/100
    # u -= 0.04
    # if u <= 0.14: return 8 + u/100
    # u -= 0.14
    # if u <= 0.12: return 9 + u/100
    # u -= 0.12
    # if u <= 0.021: return 10 + u/100
    # u -= 0.021
    # if u <= 0.007: return 11 + u/100
    # u -= 0.007
    # if u <= 0.002: return 12 + u/100
    # return 12 + u/100 # for safety
    return self.dist.rvs() + u/100

class Bern(RV):
  def __init__(self, L, U, p):
    RV.__init__(self, l_l=L, u_l=U)
    self.p = p
    
    self.v_l = [L, U]
    self.p_l = [1 - p, p]
    self.dist = scipy.stats.rv_discrete(name='bern', values=(self.v_l, self.p_l) )
  
  def __str__(self):
    return "Bern(l= {}, u= {}, p= {})".format(self.l_l, self.u_l, self.p)
  
  def pdf(self, x):
    # return (x == self.l_l)*(1 - self.p) + (x == self.u_l)*self.p
    return self.dist.pmf(x)
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return self.dist.cdf(math.floor(x) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def mean(self):
    # return (1 - self.p)*self.l_l + self.p*self.u_l
    return self.dist.mean()
  
  def moment(self, i):
    # p = 1/(self.u_l - self.l_l + 1)
    # return sum([p*v**i for v in range(self.l_l, self.u_l+1) ] )
    return self.dist.moment(i)
  
  def gen_sample(self):
    # u = random.uniform(0, 1)
    # return self.u_l + u/100 if u <= self.p else self.l_l + u/100
    return self.dist.rvs()

# class BernPareto(RV):
#   def __init__(self, U, L, p, loc, a):
#     RV.__init__(self, l_l=U*loc, u_l=float("Inf") )
    
#     self.bern = Bern(U, L, p)
#     self.pareto = Pareto(loc, a)
  
#   def __str__(self):
#     return "Bern*Pareto"
  
#   def mean(self):
#     return self.bern.mean()*self.pareto.mean()
  
#   def gen_sample(self):
#     return self.bern.gen_sample()*self.pareto.gen_sample()

class DUniform():
  def __init__(self, lb, ub):
    RV.__init__(self, l_l=lb, u_l=ub)
    
    self.v = numpy.arange(self.l_l, self.u_l+1)
    w_l = [1 for v in self.v]
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='duniform', values=(self.v, self.p) )
  
  def __str__(self):
    return "DUniform[{}, {}]".format(self.l_l, self.u_l)
  
  def mean(self):
    return (self.u_l + self.l_l)/2
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    if x < self.l_l:
      return 0
    elif x > self.u_l:
      return 1
    return self.dist.cdf(math.floor(x) )
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def moment(self, i):
    # p = 1/(self.u_l - self.l_l + 1)
    # return sum([p*v**i for v in range(self.l_l, self.u_l+1) ] )
    return self.dist.moment(i)
  
  def gen_sample(self):
    # return random.randint(self.l_l, self.u_l)
    return self.dist.rvs()

class BoundedZipf():
  def __init__(self, lb, ub, a=1):
    RV.__init__(self, l_l=lb, u_l=ub)
    self.a = a
    
    self.v = numpy.arange(self.l_l, self.u_l+1) # values
    w_l = [float(v)**(-a) for v in self.v] # self.v**(-a) # weights
    self.p = [w/sum(w_l) for w in w_l]
    self.dist = scipy.stats.rv_discrete(name='bounded_zipf', values=(self.v, self.p) )
  
  def __str__(self):
    return "BoundedZipf([{}, {}], a= {})".format(self.l_l, self.u_l, self.a)
  
  def pdf(self, x):
    return self.dist.pmf(x)
  
  def cdf(self, x):
    # if x < self.l_l: return 0
    # elif x >= self.u_l: return 1
    # else:
    #   return sum(self.p[:(x-self.l_l+1) ] )
    return self.dist.cdf(x)
  
  def inv_cdf(self, p):
    return self.dist.ppf(p)
  
  def tail(self, x):
    return 1 - self.cfd(x)
  
  def mean(self):
    # return sum([v*self.p(i) for i,v in enumerate(self.v) ] )
    return self.dist.mean()
  
  def gen_sample(self):
    return self.dist.rvs(size=1)

class Binom():
  def __init__(self, n, p):
    RV.__init__(self, l_l=0, u_l=n)
    self.n = n
    self.p = p
    
    self.dist = scipy.stats.nbinom(n, p)
  
  def __str__(self):
    return "Binom[n= {}, p= {}]".format(self.n, self.p)
  
  def pdf(self, x):
    return self.dist.pdf(x)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def gen_sample(self):
    return self.dist.rvs(size=1)

class NegBinom():
  def __init__(self, num_succ, p):
    RV.__init__(self, l_l=num_succ, u_l=float("Inf") )
    self.p = p
    
    self.dist = scipy.stats.nbinom(num_succ, p)
  
  def __str__(self):
    return "NegBinom[num_succ= {}, p= {}]".format(self.l_l, self.p)
  
  def cdf(self, x):
    return self.dist.cdf(x - self.l_l)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def gen_sample(self):
    return self.dist.rvs(size=1)

class Gamma():
  def __init__(self, num_exp, rate):
    RV.__init__(self, l_l=0, u_l=float("Inf") )
    
    self.shape, self.scale = num_exp, 1/rate
    # self.dist = numpy.random.gamma(shape, scale, size=1)
    self.dist = scipy.stats.gamma(self.shape, self.scale)
  
  def __str__(self):
    return "Gamma[shape= {}, scale= {}]".format(self.shape, self.scale)
  
  def cdf(self, x):
    return self.dist.cdf(x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def gen_sample(self):
    return self.dist.rvs(size=1)

class X_n_k():
  def __init__(self, X, n, k):
    RV.__init__(self, l_l=X.l_l, u_l=X.u_l)
    self.X, self.n, self.k = X, n, k
  
  def __str__(self):
    return "{}_{{}:{}}".format(X, self.n, self.k)
  
  def pdf(self, x):
    return self.n*self.X.pdf(x) * binomial(self.n-1, self.k-1) * self.X.cdf(x)**(self.k-1) * self.X.tail(x)**(self.n-self.k)
  
  def cdf(self, x):
    return cdf_n_k(self.X, self.n, self.k, x)
  
  def tail(self, x):
    return 1 - self.cdf(x)
  
  def moment(self, i):
    return float(mpmath.quad(lambda x: i*x**(i-1) * self.tail(x), [0, mpmath.inf] ) )
  
  def gen_sample(self):
    return gen_orderstat_sample(self.X, self.n, self.k)

def binomial(n, k):
  # if n == k:
  #   return 1
  # elif k == 1:
  #   return n
  # elif k == 0:
  #   return 1
  # elif k > n:
  #   return 0
  # else:
  #   return math.factorial(n)/math.factorial(k)/math.factorial(n-k)
  return scipy.special.binom(n, k)

def moment_ith(i, X):
  # return float(mpmath.quad(lambda x: i*x**(i-1) * X.tail(x), [0, X.u_l] ) ) # mpmath.inf 10000*10
  return float(mpmath.quad(lambda x: i*x**(i-1) * X.tail(x), [0, mpmath.inf] ) )

# Order stats
def cdf_n_k(X, n, k, x): # Pr{X_n:k < x}
  cdf = 0
  for i in range(k, n+1):
    cdf += binomial(n, i) * X.cdf(x)**i * X.tail(x)**(n-i)
  return cdf

def moment_ith_n_k(X, i, n, k): # E[X_n:k]
  return mpmath.quad(lambda x: i*x**(i-1) * (1 - cdf_n_k(X, n, k, x) ), [0, 10000*10] )

def gen_orderstat_sample(X, n, k):
  # print("s_l= {}".format(s_l) )
  return sorted([X.gen_sample() for _ in range(n) ] )[k-1]

def H(n):
  if n == 0:
    return 0
  sum_ = 0
  for i in range(1, n+1):
    sum_ += float(1/i)
  return sum_

def fact(n):
  return math.factorial(n)

def laplace(X, r):
  return mpmath.quad(lambda x: math.exp(-r*x) * X.pdf(x), [0, X.u_l] )

def inverse_laplace(F, t):
  # """
  # Tupple "a", of five complex members.
  a = 12.83767675+1.666063445j, 12.22613209+5.012718792j,\
      10.93430308+8.409673116j, 8.776434715+11.92185389j,\
      5.225453361+15.72952905j
  # Tupple "K", of five complex members.
  K = -36902.08210+196990.4257j, 61277.02524-95408.62551j,\
      -28916.56288+18169.18531j, +4655.361138-1.901528642j,\
      -118.7414011-141.3036911j

  sum = 0.0
  # Zakian's method does not work for t=0. Check that out.
  if t == 0:
    return None
  
  # The for-loop below is the heart of Zakian's Inversion Algorithm.
  for j in range(0, 5):
    sum = sum + (K[j]*F(a[j]/t) ).real
 
  return (2.0*sum/t)
  # """
  """
  # *** Function trapezoid computes the numerical inversion. ***
  def trapezoid(t, omega, sigma, nint):
    sum = 0.0
    delta = float(omega)/nint
    wi = 0.0
    
    # The for-loop below computes the FFT Inversion Algorithm.
    # It is in fact the trapezoidal rule for numerical integration.
    for i in range(1, (nint+1)):
      witi = complex(0,wi*t)

      wf = wi + delta
      wfti = complex(0,wf*t)

      fi = (exp(witi)*F(complex(sigma,wi))).real
      ff = (exp(wfti)*F(complex(sigma,wf))).real
      sum = sum + 0.5*(wf-wi)*(fi+ff)
      wi = wf
    return (sum*exp(sigma*t)/pi).real
  
  omega, sigma, nint = 200, 2.05, 1000 # 10000
  omega = math.ceil(omega)
  nint = math.ceil(nint)
  if omega <= 0:
    omega = 200
  if nint <= 0:
    nint = 10000
  
  return trapezoid(t, omega, sigma, nint)
  """

def check_invertlaplace():
  V = HyperExp([0.8, 0.2], [1, 0.1] )
  
  V_laplace = lambda s: V.laplace(s)
  def V_pdf(t):
    return mpmath.invertlaplace(V_laplace, t, method='talbot')
  
  EV = V.moment(1)
  EV_ = mpmath.quad(lambda v: v*V_pdf(v), [0, mpmath.inf] )
  # EV_ = mpmath.quad(lambda v: V.tail(v), [0, mpmath.inf] )
  print("EV= {}, EV_= {}".format(EV, EV_) )
  
  # t_l = []
  # pdf_l, invertedpdf_l = [], []
  # for t in numpy.linspace(0.01, 20, 20):
  #   t_l.append(t)
  #   pdf_l.append(V.pdf(t) )
  #   invertedpdf_l.append(V_pdf(t) )
  # plot.plot(t_l, pdf_l, label='Pdf', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  # plot.plot(t_l, invertedpdf_l, label='Inverted Pdf', color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  # plot.xlabel(r'$t$', fontsize=14)
  # plot.ylabel(r'PDF', fontsize=14)
  # plot.legend()
  # plot.savefig("check_invertlaplace.pdf")
  # plot.gcf().clear()
  alog("done.")

def check_tpareto():
  rv = TPareto(1, 1000, 1.1)
  s_l = []
  for i in range(10000*2):
    s_l.append(rv.gen_sample() )
  
  s_l = numpy.sort(s_l)[::-1]
  y_l = numpy.arange(s_l.size)/s_l.size
  plot.plot(s_l, y_l, label=r'${}$'.format(rv), marker=next(marker), color=next(dark_color), linestyle='-', mew=mew, ms=ms)
  
  plot.xscale('log')
  plot.yscale('log')
  plot.xlabel(r'$x$', fontsize=14)
  plot.ylabel(r'$Pr\{X > x\}$', fontsize=14)
  plot.legend()
  plot.savefig("check_tpareto.pdf")
  plot.gcf().clear()
  log(WARNING, "done.")

if __name__ == "__main__":
  # plot_gensample_check()
  
  # D = Dolly()
  # # print("Dolly sample= {}".format(D.gen_sample() ) )
  # x_l, cdf_l = [], []
  # for x in numpy.linspace(0, 20, 100):
  #   x_l.append(x)
  #   cdf_l.append(D.cdf(x) )
  # plot.plot(x_l, cdf_l, label=r'CDF of Dolly', marker=next(marker), linestyle=':', mew=2)
  # plot.legend()
  # plot.savefig("plot_dolly_cdf.pdf")
  # plot.gcf().clear()
  
  # check_invertlaplace()
  
  D = Dolly()
  print("ED= {}".format(D.mean() ) )
  V = TPareto(1, 100, 1.1)
  print("EV= {}".format(V.mean() ) )
