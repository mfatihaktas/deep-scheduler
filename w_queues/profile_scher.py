import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import numpy as np

from scher import PolicyGradScher
from patch import *
import learn_howtorep_wmpi as lhw

# ar_l = [*np.linspace(0.01, 0.9*lhw.ar_ub, 4, endpoint=False), *np.linspace(0.9*lhw.ar_ub, lhw.ar_ub, 4) ]
ar_l = [0.17]

def plot_probrep():
  for ar in ar_l: # lhw.ar_l
    scher = PolicyGradScher(lhw.s_len, lhw.a_len, lhw.nn_len, save_name=save_name('saved', 'howtorep', lhw.ns, lhw.d, ar) )
    scher.restore(lhw.L)
    
    max_l = int(lhw.L/2)
    grid = np.zeros((max_l, max_l))
    for l1 in range(max_l):
      for l2 in range(l1, max_l):
        p = scher.get_action_dist([l1, l2] )
        grid[l1, l2] = p[1]
    # print("grid= \n{}".format(grid) )
    # extent = [0.5, n+0.5, 0.5, n+0.5]
    # img = plot.imshow(grid, cmap='gray_r', extent=extent, origin='lower')
    img = plot.imshow(grid, cmap='gray_r', origin='lower')
    plot.colorbar(img, cmap='gray_r')
    plot.title(r'$n= {}$, $d= {}$, $\lambda= {}$'.format(lhw.ns, lhw.d, "{0:.2f}".format(ar) ) + "\n" + '$J \sim {}$, $S \sim {}$'.format(lhw.J.tolatex(), lhw.S.tolatex() ) )
    # plot.xticks(range(max_l), fontsize=12)
    plot.xlabel(r'$l_1$', fontsize=14)
    # plot.yticks(range(max_l), fontsize=12)
    plot.ylabel(r'$l_2$', fontsize=14)
    
    plot.legend()
    plot.savefig("plot_probrep_ar{0:.2f}.png".format(ar) )
    log(WARNING, "done; ar= {}".format(ar) )
    plot.gcf().clear()
  log(WARNING, "done.")

def plot_probrep_wrt_jsize(ar):
  scher = PolicyGradScher(s_len, a_len, nn_len, save_name=save_name('saved', 'howtorep', ns, d, ar) )
  scher.restore(lhw.L)
  
  js_l, probrep_l = [], []
  for js in np.linspace(lhw.J.l_l, lhw.u_l, 100):
    js_l.append(js)
    p = scher.get_action_dist([0]*lhw.d + [js] )
    probrep_l.append(p[1] )
  plot.plot(js_l, probrep_l, color=next(dark_color), marker=next(marker), linestyle=':', mew=2)
  
  plot.title(r'$J= {}$'.format(lhw.J) )
  plot.xlabel('Job size', fontsize=14)
  plot.ylabel('Prob of rep', fontsize=14)
  plot.legend()
  plot.savefig("plot_probrep_wrt_jsize_ar{}.pdf".format(ar) )
  plot.gcf().clear()
  alog("done; ar= {}".format(ar) )

def plot_scher(i, scher, J):
  def plot_aprob_heatmap(J):
    max_ql = 100
    grid = numpy.zeros((max_ql, max_ql))
    for ql1 in range(max_ql):
      for ql2 in range(ql1, max_ql):
        p = scher.get_action_dist([ql1, ql2, J] )
        grid[ql1, ql2] = p[1]
    img = plot.imshow(grid, cmap='gray_r', origin='lower')
    plot.colorbar(img, cmap='gray_r')
    
    plot.title(r'$J= {0:.2f}$'.format(J) )
    plot.xlabel(r'$l_2$', fontsize=14)
    plot.ylabel(r'$l_1$', fontsize=14)
    plot.legend()
  plot.subplot(1, 3, 1)
  plot_aprob_heatmap(J.l_l)
  plot.subplot(1, 3, 2)
  plot_aprob_heatmap(J.mean() )
  plot.subplot(1, 3, 3)
  plot_aprob_heatmap(J.u_l/2)
  
  plot.tight_layout()
  plot.savefig("plot_aprob_heatmap_{}.png".format(i) )
  plot.gcf().clear()
  log(WARNING, "done; i= {}.".format(i) )

if __name__ == "__main__":
  plot_probrep()
