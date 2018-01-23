import inspect, math, mpmath, scipy, itertools
from scipy import special

# dark_color = itertools.cycle(('green', 'red', 'blue', 'turquoise', 'goldenrod', 'purple', 'gray', 'brown', 'magenta', 'gold', 'olive', 'orangered', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray'))
dark_color = itertools.cycle(('green', 'red', 'blue', 'goldenrod', 'magenta', 'purple', 'gray', 'brown', 'turquoise', 'gold', 'olive', 'silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'darkgray', 'orangered'))
light_color = itertools.cycle(('silver', 'rosybrown', 'plum', 'lightsteelblue', 'lightpink', 'orange', 'turquoise'))
linestyle = itertools.cycle(('-', '--', '-.', ':') )
marker = itertools.cycle(('^', 'p', 'd', '+', 'v', '<', '>', '1' , '2', '3', '4', 'x') )
skinny_marker_l = ['x', '+', '1', '2', '3', '4']

mew, ms = 3, 5

INFO = 0
DEBUG = 1
WARNING = 2
ERROR = 3

# DEBUG_LEVEL = INFO
DEBUG_LEVEL = WARNING
# DEBUG_LEVEL = ERROR

debug_level__string_map = {INFO: "INFO", DEBUG: "DEBUG", WARNING: "WARNING", ERROR: "ERROR"}

"""
*log: To have a unified logging which can be refactored easily
"""
def sim_log(dlevel, env, caller, action, affected):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  env= simpy.Environment
  caller= string -- name of the sim component acting
  action= string
  affected= any -- whatever component being acted on/with e.g., packet
  """
  if DEBUG_LEVEL <= dlevel:
    print("{} t: {:.2f}] {} {}\n\t{}".format(debug_level__string_map[dlevel], env.now, caller, action, affected) )

def log(dlevel, log):
  """
  Parameters
  ----------
  dlevel= int -- debug level
  log= string to be logged
  """
  if DEBUG_LEVEL <= dlevel:
    print("{}] {}:: {}".format(debug_level__string_map[dlevel], inspect.stack()[1][3], log) )

def list_to_str(l):
  return ",".join("%s" % e for e in l)

def H_cont(n):
  return mpmath.quad(lambda x: (1-x**n)/(1-x), [0, 1] )

def H(n):
  if n == 0:
    return 0
  sum_ = 0
  for i in range(1, n+1):
    sum_ += float(1/i)
  return sum_

def H_2(n):
  sum_ = 0
  for i in range(1, n+1):
    sum_ += float(1/(i**2) )
  return sum_

def gen_H(n, k):
  sum_ = 0
  for i in range(1, n+1):
    if (i - k) == 0:
      continue
    sum_ += float(1/(i*(i - k) ) )
  return sum_

def binomial(n, k):
  if n == k:
    return 1
  elif k == 1:
    return n
  elif k == 0:
    return 1
  elif k > n:
    return 0
  else:
    return math.factorial(n)/math.factorial(k)/math.factorial(n-k)

# def binomial(x, y):
#   try:
#     binom = factorial(x) // factorial(y) // factorial(x - y)
#   except ValueError:
#     binom = 0
#   return binom

def I(u_l, m, n):
  den = B(m, n)
  if den == 0:
    return None
  return B(m, n, u_l=u_l)/den
  # return scipy.special.betainc(m, n, u_l)

def B(m, n, u_l=1):
  # if u_l == 1:
  #   return scipy.special.beta(m, n)
  return mpmath.quad(lambda x: x**(m-1) * (1-x)**(n-1), [0, u_l] )
  # else:
  #   return I(u_l, m, n)*B(m, n)

def G(z):
  return scipy.special.gamma(z)
  # return mpmath.quad(lambda x: x**(z-1) * math.exp(-z), [0, mpmath.inf] )

# Order stats
def cdf_n_k(n, k, X, x): # Pr{X_n:k < x}
  cdf = 0
  for i in range(k, n+1):
    cdf += binomial(n, i) * X.cdf(x)**i * X.tail(x)**(n-i)
  return cdf

def moment_i_n_k(i, n, k, X): # E[X_n:k]
  return mpmath.quad(lambda x: i*x**(i-1) * (1 - cdf_n_k(n, k, X, x) ), [0, mpmath.inf] )

# Qing
def PK(E_V, E_V_2, ar):
  if ar*E_V >= 1:
    return None
  E_T = E_V + ar*E_V_2/2/(1 - ar*E_V)
  if E_T > 100: return None
  return E_T

def fit_pareto(s_l):
  n = len(s_l)
  
  fit_upper_tail = False # True
  if not fit_upper_tail:
    l = s_l[-1]
    D = 0
    for s in s_l:
      D += math.log(s) - math.log(l)
    a = (n-1)/D
  elif fit_upper_tail:
    l = s_l[-1]
    i = int(math.sqrt(n) ) # int(n*0.3)
    s_l = s_l[:i]
    l_ = s_l[-1]
    D = 0
    for s in s_l:
      D += math.log(s) - math.log(l_)
    a = i/D
  # log(WARNING, "done; l= {}, a= {}".format(l, a) )
  return l, a

def fit_tpareto(s_l):
  # s_l is ordered in descending order
  n = len(s_l)
  log(WARNING, "n= {}".format(n) )
  fit_upper_tail = False # True
  def solve_a(eq):
    a = 0.01
    _a = None
    while True:
      if eq(a) > 0:
        _a = a
        a += 0.01
      else:
        return _a if _a is not None else 0.01
  
  u = s_l[0]
  if not fit_upper_tail:
    l = s_l[-1]
    r = l/u
    # Did not work somehow
    # a = sympy.Symbol('a')
    # a = sympy.solve(n/a + n*r**a*math.log(r)/(1-r**a) - sum([math.log(x/l) for x in s_l] ) )
    a = solve_a(lambda a: n/a + n*r**a*math.log(r)/(1-r**a) - sum([math.log(x/l) for x in s_l] ) )
  else:
    i = int(math.sqrt(n) ) # int(n*0.3)
    X_ip1 = s_l[i+1]
    r = X_ip1/u
    a = solve_a(lambda a: i/a + i*r**a*math.log(r)/(1-r**a) - sum([math.log(x) - math.log(X_ip1) for x in s_l[:i+1] ] ) )
    l = i**(1/a) * X_ip1*(n - (n-i)*(X_ip1/u)**a)**(-1/a)
  log(WARNING, "done; l= {}, u= {}, a= {}".format(l, u, a) )
  return l, u, a

def fit_sexp(s_l):
  # https://www.statlect.com/fundamentals-of-statistics/exponential-distribution-maximum-likelihood
  D = min(s_l)
  n = len(s_l)
  mu = n/(sum(s_l) - n*D)
  
  return D, mu
