


def ET_wrelaunch_pareto(a, alpha, d, k, n):
  # ET_k_n_pareto(i, i, a, alpha)
  
  def g(k, alpha):
    if k > 170:
      return a*(k+1)**(1/alpha) * G(1-1/alpha)
    return a*G(1-1/alpha)*G(k+1)/G(k+1-1/alpha)
    
    return d*(1-q**k) + g(k, alpha)*((a/d-1)*I(1-q, 1-1/alpha, k) + 1)