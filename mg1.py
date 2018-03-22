import numpy as np
import cmath

from sim import *
from patch import *
from rvs import *

ET_MAX = 20000

def ar_ub_mg1(J, S):
  return 1/J.mean()/S.mean()

def ET_mg1(ar, J, S):
  EB = J.mean()*S.mean()
  EB2 = moment_ith(2, J)*moment_ith(2, S)
  ET = EB + ar*EB2/2/(1 - ar*EB)
  return ET if ET < ET_MAX else None

def plot_mg1():
  J = Exp(1) # TPareto(1, 10, 1.1)
  S = DUniform(1, 1) # TPareto(1, 12, 3)
  ar_ub = ar_ub_mg1(J, S)
  alog("J= {}, S= {}, ar_ub= {}".format(J, S, ar_ub) )
  
  T = 25000
  def sim(ar):
    env = simpy.Environment()
    jg = JG(env, ar, DUniform(1, 1), J, T)
    q = FCFS(0, env, S)
    jg.out = q
    jg.init()
    env.run()
    
    return np.mean(q.lt_l)
  
  ar_l, ET_l = [], []
  for ar in np.linspace(0.05, ar_ub + 0.1, 2):
    ar_l.append(ar)
    print("> ar= {}".format(ar) )
    ET = sim(ar)
    ETm = ET_mg1(ar, J, S)
    print("ET= {}, ETm= {}".format(ET, ETm) )

def laplace_inverse():
  ar = 0.5
  V = Exp(1) # TPareto(1, 10**3, 2)
  EV = V.mean()
  ro = ar*EV
  
  # def T_laplace(s):
  #   V_laplace = laplace(V, s)
  #   return (1 - ro)*s/(s - ar + ar*V_laplace)*V_laplace
  
  # def T_pdf(t):
  #   return inverse_laplace(T_laplace, t)
  
  def complex_laplace(X, s):
    def real(x):
      try:
        return scipy.real(cmath.exp(-s*x) * X.pdf(x) )
      except OverflowError:
        return float('inf')
    def imag(x):
      try:
        return scipy.imag(cmath.exp(-s*x) * X.pdf(x) )
      except OverflowError:
        return float('inf')
    
    real_integral = scipy.integrate.quad(real, X.l_l, X.u_l)
    imag_integral = scipy.integrate.quad(imag, X.l_l, X.u_l)
    return real_integral[0] + 1j*imag_integral[0]
  
  V_laplace = lambda s: complex_laplace(V, s) # 1/(s + 1)
  def V_pdf(t):
    # return inverse_laplace(V_laplace, t)
    return mpmath.invertlaplace(V_laplace, t, method='talbot')
  
  for t in np.linspace(0.05, 10, 20): # V.u_l
    actual_pdf = V.pdf(t)
    inverted_pdf = V_pdf(t)
    print("actual_pdf= {}, inverted_pdf= {}".format(actual_pdf, inverted_pdf) )

if __name__ == "__main__":
  # plot_mg1()
  laplace_inverse()
