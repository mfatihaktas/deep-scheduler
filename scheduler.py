import numpy as np
import concurrent.futures
from operator import itemgetter

from sim_objs import *
from sim_exp import arrival_rate_upperbound
from rlearning import *

# ############################################  Scher  ########################################### #
class Scher(object):
  def __init__(self, sching_m):
    self.sching_m = sching_m
    
    self.s_len = 1
    
    if sching_m['type'] == 'plain':
      self.schedule = self.plain
    elif sching_m['type'] == 'expand_if_totaldemand_leq':
      self.schedule = self.expand_if_totaldemand_leq
  
  def __repr__(self):
    return "Scher[sching_m={}]".format(self.sching_m)
  
  def plain(self, job, wload_l):
    return None, self.sching_m['a']
  
  def expand_if_totaldemand_leq(self, job, wload_l):
    a = self.sching_m['a'] if job.totaldemand < self.sching_m['threshold'] else 0
    return None, a

# ###########################################  RLScher  ########################################## #
def state(job, wload_l):
  # s = [job.k, job.totaldemand, min(wload_l), max(wload_l), np.mean(wload_l), np.std(wload_l) ]
  s = [job.k, job.totaldemand]
  return s

class RLScher(Scher):
  def __init__(self, sinfo_m, sching_m):
    self.sinfo_m = sinfo_m
    
    self.s_len = 2
    self.a_len = 2 # expansion rate: 1, 2
    self.N, self.T = sching_m['N'], sinfo_m['njob']
    
    self.learner = PolicyGradLearner(s_len=self.s_len, a_len=self.a_len, nn_len=10, w_actorcritic=True)
  
  def __repr__(self):
    return "RLScher[learner=\n{}]".format(self.learner)
  
  def save(self, i, save_name=None):
    return self.learner.save(i, save_name)
  
  def restore(self, i, save_name=None):
    return self.learner.restore(i, save_name)
  
  def schedule(self, job, wload_l):
    s = state(job, wload_l)
    a = self.learner.get_random_action(s)
    # if a < 1:
    #   a = 1
    # elif int(a*job.k) > len(wload_l):
    #   a = len(wload_l)/job.k
    return s, a
  
  def train(self, nsteps):
    for i in range(nsteps):
      alog(">> i= {}".format(i) )
      n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((self.N, self.T, self.s_len)), np.zeros((self.N, self.T, 1)), np.zeros((self.N, self.T, 1))
      for n in range(self.N):
        t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(self.sinfo_m, self)
        alog("n= {}, avg_a= {}, avg_r= {}, avg_sl= {}".format(n, np.mean(t_a_l), np.mean(t_r_l), np.mean(t_sl_l) ) )
        n_t_s_l[n], n_t_a_l[n], n_t_r_l[n] = t_s_l, t_a_l, t_r_l
      self.learner.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
  
  def train_multithreaded(self, nsteps):
    if self.learner.restore(nsteps):
      log(WARNING, "learner.restore is a success, will not retrain.")
    else:
      tp = concurrent.futures.ThreadPoolExecutor(max_workers=100)
      for i in range(1, nsteps+1):
        alog(">> i= {}".format(i) )
        n_t_s_l, n_t_a_l, n_t_r_l = np.zeros((self.N, self.T, self.s_len)), np.zeros((self.N, self.T, 1)), np.zeros((self.N, self.T, 1))
        future_n_m = {tp.submit(sample_traj, self.sinfo_m, self): n for n in range(self.N) }
        for future in concurrent.futures.as_completed(future_n_m):
          n = future_n_m[future]
          try:
            t_s_l, t_a_l, t_r_l, t_sl_l = future.result()
          except Exception as exc:
            log(ERROR, "exception;", exc=exc)
          alog("n= {}, avg_a= {}, avg_r= {}, avg_sl= {}".format(n, np.mean(t_a_l), np.mean(t_r_l), np.mean(t_sl_l) ) )
          n_t_s_l[n], n_t_a_l[n], n_t_r_l[n] = t_s_l, t_a_l, t_r_l
        self.learner.train_w_mult_trajs(n_t_s_l, n_t_a_l, n_t_r_l)
        self.learner.save(i)
  
  def plot(self):
    load_l, Pr_bind_l = [], []
    for l in np.linspace(0, 1, 100):
      load_l.append(l)
      
      p = self.learner.get_action_dist([l] )[1]
      print("l= {}, p= {}".format(l, p) )
      Pr_bind_l.append(p)
    plot.plot(load_l, Pr_bind_l, color=NICE_RED, marker='.', linestyle=':', mew=2)
    
    # plot.legend()
    plot.xlabel('Worker load', fontsize=14)
    plot.ylabel('Probability of\nplacing an arrival', fontsize=14)
    # plot.title(r'$\mu= {}$, $r= {}$'.format(mu, r) )
    fig = plot.gcf()
    fig.set_size_inches(4, 3)
    plot.savefig('plot_rlscheduler.png', bbox_inches='tight')
    fig.clear()
    log(WARNING, "done.")

# ############################################  utils  ########################################### #
def sample_traj(sinfo_m, scher):
  def reward(slowdown):
    return 1/slowdown
  
  env = simpy.Environment()
  cl = Cluster(env, mapper=Mapper({'type': 'spreading'}), scher=scher, **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  
  T = sinfo_m['njob']
  t_s_l, t_a_l, t_r_l, t_sl_l = np.zeros((T, scher.s_len)), np.zeros((T, 1)), np.zeros((T, 1)), np.zeros((T, 1))
  
  t = 0
  for jid, jinfo_m in sorted(cl.jid_info_m.items(), key=itemgetter(0) ):
    # blog(t=t, jid=jid, jinfo_m=jinfo_m)
    if 'fate' in jinfo_m and jinfo_m['fate'] == 'finished':
      t_s_l[t, :] = jinfo_m['s']
      t_a_l[t, :] = jinfo_m['a']
      sl = jinfo_m['runtime']/jinfo_m['expected_lifetime']
      t_r_l[t, :] = reward(sl)
      t_sl_l[t, :] = sl
      t += 1
  return t_s_l, t_a_l, t_r_l, t_sl_l

def evaluate(sinfo_m, scher):
  alog("scher= {}".format(scher) )
  for _ in range(3):
    t_s_l, t_a_l, t_r_l, t_sl_l = sample_traj(sinfo_m, scher)
    print("avg_s= {}, avg_a= {}, avg_r= {}".format(np.mean(t_s_l), np.mean(t_a_l), np.mean(t_r_l) ) )

if __name__ == '__main__':
  sinfo_m = {
    'njob': 10000, 'nworker': 10, 'wcap': 10, # 10000
    'totaldemand_rv': TPareto(1, 10000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 10, 1.1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown_rv': Uniform(0.1, 0.5),
      'straggle_dur_rv': TPareto(1, 100, 1.1),
      'normal_dur_rv': TPareto(1, 100, 1.1) } }
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 3/4*ar_ub
  sching_m = {'N': 10}
  blog(sinfo_m=sinfo_m, sching_m=sching_m)
  
  scher = RLScher(sinfo_m, sching_m)
  # sinfo_m['max_exprate'] = max_exprate
  
  print("scher= {}".format(scher) )
  scher.train_multithreaded(40) # train(40)
  evaluate(sinfo_m, scher)
