from scheduler import *
from plot_utils import *

def plot_Pr_rep(step, s_len, a_len, nn_len, job_totaldemand_rv):
  learner = QLearner(s_len, a_len, nn_len)
  if not learner.restore(step):
    log(ERROR, "learner.restore failed!")
    return
  
  load1_l, load2_l, totaldemand_l = [], [], []
  a_l = []
  for load1 in np.linspace(0, 1, 10):
    for load2 in np.linspace(0, 1, 10):
      for totaldemand in np.logspace(0.1, math.log10(job_totaldemand_rv.u_l), 10):
      # for totaldemand in np.logspace(0.01, math.log10(job_totaldemand_rv.mean() ), 10):
      # for totaldemand in np.linspace(0.1, 100, 10):
        load1_l.append(load1)
        load2_l.append(load2)
        totaldemand_l.append(totaldemand)
        
        j = Job(_id=0, k=1, n=1, demandperslot_rv=TNormal(1, 1), totaldemand=totaldemand)
        a = learner.get_max_action(state(j, [load1, load2] ) )
        print("load1= {}, load2= {}, totaldemand= {}, a= {}".format(load1, load2, totaldemand, a) )
        a_l.append(a)

if __name__ == "__main__":
  plot_Pr_rep(step=50, s_len=3, a_len=2, nn_len=10, job_totaldemand_rv=TPareto(10, 10000, 1.1) )
  
  '''
  # x and y and z coordinates
  x = np.array(range(10))
  y = np.array(range(10,15))
  z = np.array(range(15,20))
  data_value = np.random.randint(1, 4, size=(len(x), len(y), len(z) ) )
  
  fig = plot.figure(figsize=(10,4))
  ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
  ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])
  ax.set_aspect('equal')
  
  plot_matrix(ax, x, y, z, data_value, cmap="jet", cax = ax_cb)

  plot.savefig('deneme.png')
  '''
  