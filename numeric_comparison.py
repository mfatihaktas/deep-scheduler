import numpy as np

from math_utils import *
from log_utils import *

def exact_vs_approx():
  exact = lambda k, n, a: G(n+1)/G(n+1-1/a)*G(n-k+1-1/a)/G(n-k+1)
  approx = lambda k, n, a: (1 - k/n)**(-1/a)
  
  def print_():
    s = ''
    for k in np.arange(6, 20, 2):
      n_l = np.arange(k+1, 2*k, 2)
      s += '\multirow{' + str(len(n_l)) + '}{*}{' + str(k) + '}'
      for n in n_l:
        s += ' & {}'.format(n) if n == k+1 else '  & {}'.format(n)
        for a in np.arange(2, 10, 1):
          exact_v = exact(k, n, a)
          approx_v = approx(k, n, a)
          approx_err = round(100*abs(exact_v-approx_v)/exact_v, 2)
          # print("k= {}, n= {}, a= {}, approx_err= {}".format(k, n, a, approx_err) )
          s += ' & {}'.format(approx_err)
        s += ' \\\\ \n'
      s += '\hline\n'
    print(s)
  
  def plot_():
    k_n__exact_approx_l_m = {}
    a_l = np.linspace(2, 10, 10)
    for k in np.arange(10, 20, 1):
      for n in np.arange(k, 2*k, 3):
        exact_l, approx_l = [], []
        for a in a_l:
          exact_l.append(exact(k, n, a) )
          approx_l.append(approx(k, n, a) )
        k_n__exact_approx_l_m[(k, n)] = {'exact_l': exact_l, 'approx_l': approx_l}
    
    nrows = len(k_n__exact_approx_l_m)
    fig, axs = plot.subplots(nrows, 1, sharex='col')
    figsize = [4, nrows*3]
    i = 0
    for (k, n), y_m in k_n__exact_approx_l_m.items():
      ax = axs[i]
      plot.sca(ax)
      
      plot.plot(a_l, y_m['exact_l'], label='Exact', color='black', marker='.', linestyle='-', mew=2, ms=5)
      plot.plot(a_l, y_m['approx_l'], label='Approx', color='blue', marker='x', linestyle=':', mew=2, ms=5)
      
      plot.ylabel(r'$\alpha$', fontsize=14)
      plot.title('k= {}, n= {}'.format(k, n), fontsize=14)
      
      i += 1
    fig.set_size_inches(figsize[0], figsize[1] )
    plot.subplots_adjust(wspace=0.25) # , hspace=0.1
    fig.patch.set_alpha(0.5)
    plot.savefig('plot_exact_vs_approx.png', bbox_inches='tight')
    fig.clear()
  
  print_()
  # plot_()
  
  log(WARNING, "done.")
  

if __name__ == "__main__":
  exact_vs_approx()