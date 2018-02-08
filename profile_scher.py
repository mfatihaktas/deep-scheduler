from learn_shortestq import *
from learn_howtorep import *
# from patch import *

def plot_probrep_d2():
  ns, d = 10, 2
  ar = 0.115 # 0.30500000000000005 # 0.4
  s_len, a_len, nn_len = d, d, 10
  
  scher = PolicyGradScher(s_len, a_len, nn_len, save_name=save_name('saved', 'howtorep', ns, d, ar) )
  scher.restore(99)
  
  max_l = 25
  l1_l2_grid = numpy.zeros((max_l, max_l))
  for l1 in range(max_l):
    for l2 in range(max_l):
      s = sorted([l1, l2] )
      p = scher.get_action_dist(s)
      l1_l2_grid[l1, l2] = p[1]
  print("l1_l2_grid= \n{}".format(l1_l2_grid) )
  # extent = [0.5, n+0.5, 0.5, n+0.5]
  # img = plot.imshow(l1_l2_grid, cmap='gray_r', extent=extent, origin='lower')
  img = plot.imshow(l1_l2_grid, cmap='gray_r', origin='lower')
  plot.colorbar(img, cmap='gray_r')
  
  plot.title(r'$n= {}$, $d= {}$, $\lambda= {}$'.format(ns, d, ar) )
  # plot.xticks(range(max_l), fontsize=12)
  plot.xlabel(r'$l_1$', fontsize=14)
  # plot.yticks(range(max_l), fontsize=12)
  plot.ylabel(r'$l_2$', fontsize=14)
  
  plot.legend()
  plot.savefig("plot_ar_{}.pdf".format(ar) )
  log(WARNING, "done.")

if __name__ == "__main__":
  plot_probrep_d2()