from rvs import *
from patch import *

def rep_wcancel():
  def Pr_Tgt(V, X, c, t):
    Pr_V_l_X = mpmath.quad(lambda x: V.cdf(x)*X.pdf(x), [0, 10000*10] )
    return V.tail(t) * (V.tail(t)*Pr_V_l_X + 1 - Pr_V_l_X)**c
  
  def momenti(i, V, X, c):
    return mpmath.quad(lambda t: i*t**(i-1) * Pr_Tgt(V, X, c, t), [0, 10000*10] ) # mpmath.inf
  
  def steadystate_Pr_Tgt(V, X, c, t):
    def Pr_Tgt_(Pr_Vgt, X, c, t):
      return  Pr_Vgt * (Pr_Vgt*X.tail(t) + X.cdf(t) )**c
    tail_pr = V.tail(t)
    for i in range(20):
      print("i= {}, tail_pr= {}".format(i, tail_pr) )
      tail_pr = Pr_Tgt_(tail_pr, X, c, t)
  
  V = Pareto(1, 2)
  def plot_tail(c, ar):
    X = Exp(ar)
    
    t_l, Pr_Tgt_l = [], []
    for t in numpy.linspace(0.05, 20, 100):
      t_l.append(t)
      Pr_Tgt_l.append(Pr_Tgt(V, X, c, t) )
    plot.plot(t_l, Pr_Tgt_l, label=r'$c= {}$, $\lambda= {}$'.format(c, ar), color=next(dark_color), linestyle='-')
    plot.xlabel(r'$t$', fontsize=14)
    plot.ylabel(r'$Pr\{T > t\}$', fontsize=14)
  
  def plot_ETi(c, i):
    ar_l, ETi_l = [], []
    for ar in numpy.logspace(-10, 0.5, 50):
      X = Exp(ar)
      ar_l.append(ar)
      ETi_l.append(momenti(i, V, X, c) )
    plot.plot(ar_l, ETi_l, label=r'$c= {}$'.format(c), color=next(dark_color), linestyle='-')
    plot.xlabel(r'$\lambda$', fontsize=14)
    plot.ylabel(r'$E[T^{}]$'.format(i), fontsize=14)
    
  c = 2
  plot_tail(c, ar=0.1)
  plot_tail(c, ar=1)
  plot_tail(c, ar=10)
  plot_tail(c, ar=100)
  
  # X = Exp(0.1)
  # steadystate_Pr_Tgt(V, X, c=1, t=2)
  
  # i = 3
  # plot_ETi(c=1, i=i)
  # plot_ETi(c=2, i=i)
  # plot_ETi(c=3, i=i)
  
  plot.legend()
  plot.title(r'$V \sim {}$, $X \sim Exp(\lambda)$'.format(V) )
  plot.savefig("rep_wcancel.png", bbox_inches='tight')
  plot.gcf().clear()
  log(WARNING, "done.")

def waitingtime_repwcancel():
  def laplace(X, r):
    return mpmath.quad(lambda x: math.exp(-r*x) * X.pdf(x), [0, X.u_l] ) # mpmath.inf 10000*10
  
  # V = Exp(1)
  # V = Pareto(1, 3)
  # V = Pareto(0.1, 3)
  V = TPareto(1, 10, 1)
  V21 = X_n_k(V, 2, 1)
  EV = moment_ith(1, V)
  EV21 = moment_ith(1, V21)
  print("EV= {}, EV21= {}".format(EV, EV21) )
  def solvefor_War(ar):
    X = Exp(ar)
    V_21 = X_n_k(V, 2, 1)
    a = laplace(V_21, ar)
    b = laplace(V, ar)
    print("a= {}, b= {}".format(a, b) )
    
    ro = ar*EV
    eq = lambda W: (a-b)*W**2 + (b + ar*(EV21 - EV) )*W + ar*EV-1
    l, u = 0.0001, 2
    print("eq(l)= {}, eq(u)= {}".format(eq(l), eq(u) ) )
    roots = scipy.optimize.brentq(eq, l, u)
    print("ar= {}, roots= {}".format(ar, roots) )

  # for ar in numpy.linspace(0.05, 1/EV-0.05, 10):
  for ar in numpy.linspace(0.0001, 1/EV-0.05, 10):
    solvefor_War(ar)



if __name__ == "__main__":
  # waitingtime_repwcancel()
  pass
