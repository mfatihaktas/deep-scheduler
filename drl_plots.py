import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from experience_replay import *

ro_l = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

R = sinfo_m['reqed_rv']
L = sinfo_m['lifetime_rv']
l = R.l_l*L.l_l
u = 1000*l
i = u/10
logl, logi, logu = math.log10(l), math.log10(i), math.log10(u)
# D_l = list(np.logspace(logl, logi, 50, endpoint=False) ) + list(np.logspace(logi, logu, 50) )
D_l = np.logspace(logl, logu, 50)

m_l = ['o', '^', 'v', 'd']
c_l = [NICE_BLUE, NICE_GREEN, NICE_ORANGE, NICE_RED]

scher = RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay_persist')

def plot_learned_policy():
  ax = plot.axes(projection='3d')
  ## For STATE_LEN = 3
  for ro in ro_l:
    if not scher.restore(ro__learning_count_m[ro] ):
      continue
    
    for qlen in range(10):
      for D in D_l:
        a_ = scher.learner.get_max_action(state_(D, [ro], qlen) )
        ax.plot3D([ro], [qlen], [D], c=c_l[a_], marker=m_l[a_] )
  
  ax.legend([Line2D([0], [0], c=c, ls=':', lw=1) for c in c_l], ['+{}'.format(a) for a in range(sching_m['a']+1) ] )
  plot.legend()
  fontsize = 18
  ax.set_xlabel('Average load', fontsize=fontsize)
  ax.set_ylabel('Job queue length', fontsize=fontsize)
  ax.set_zlabel('Job demand', fontsize=fontsize)
  # ax.view_init(20, 30)
  # plot.title(r'$k= {}$, $m= {}$, $C= {}$'.format(self.k, self.m, self.C), fontsize=fontsize)
  plot.savefig('plot_learned_policy.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")

def plot_action_vs_ro_demand():
  ro = 0.7
  if not scher.restore(ro__learning_count_m[ro] ):
    return
  scher.summarize()
  
  # for ro in ro_l:
  for Eload in np.linspace(0.1, 0.9, 10):
    for D in D_l:
      a_ = scher.learner.get_max_action(state_(D, [Eload]) )
      plot.plot([D], [Eload], c=c_l[a_], marker=m_l[a_] )
  plot.gca().legend(
    [Line2D([0], [0], c=c_l[a], marker=m_l[a], ls=':', lw=1) for a in range(sching_m['a']+1) ],
    ['+{}'.format(a) for a in range(sching_m['a']+1) ] )
  prettify(plot.gca() )
  plot.legend(framealpha=0.5)
  plot.xscale('log')
  fontsize = 14
  plot.xlabel('Job demand', fontsize=fontsize)
  plot.ylabel('Average load', fontsize=fontsize)
  plot.title(r'$\rho= {}$'.format(ro), fontsize=fontsize)
  plot.savefig('plot_action_vs_ro_demand.png', bbox_inches='tight')
  plot.gcf().clear()
  log(INFO, "done.")
  
def plot_Esl_vs_ro():
  ro_l = []
  

RedNone_ro__Esl_l_m = {}
RedAll_ro__Esl_l_m = {
  0.1: [],
  0.2: [],
  0.3: [],
  0.4: [],
  0.5: [],
  0.6: [],
  0.7: [],
  0.8: [],
  0.9: [] }
RedDRL_ro__Esl_l_m = {}

if __name__ == "__main__":
  # plot_learned_policy()
  plot_action_vs_ro_demand()
