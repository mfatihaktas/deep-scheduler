import random, sys, getopt

from sim_objs import *
from scheduler import *
from modeling import *

# ###################################  Cluster_wExpReplay  ####################################### #
use_lessreal_sim = True # False
class Cluster_wExpReplay(Cluster_LessReal if use_lessreal_sim else Cluster):
  def __init__(self, env, nworker, wcap, straggle_m, scher, M, **kwargs):
    super().__init__(env, float('Inf'), nworker, wcap, straggle_m, scher)
    self.M = M # number of (s, a, r, snext, anext) to collect per training
    
    # self.waitfor_jid_l = []
    self.waitforjid_begin = 5001 # 1
    self.waitforjid_end = 5000 + M # M
    self.waitfor_njob = M
    self.last_sched_jid = None
    
    self.learning_count = 0
    
    self.Er_l = []
    self.Esl_l = []
    self.loss_l = []
  
  def __repr__(self):
    return super() + '_wExpReplay!'
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      while True:
        j.wait_time = self.env.now - j.arrival_time
        s, a, w_l = self.scher.schedule(j, self.w_l, self)
        if a == -1:
          yield self.env.timeout(0.1)
        else:
          break
      
      self.last_sched_jid = j._id
      self.jid_info_m[j._id] = {'wait_time': self.env.now - j.arrival_time}
      wid_l = []
      for i, w in enumerate(w_l):
        type_ = 's' if i+1 <= j.k else 'r'
        if use_lessreal_sim:
          w.put(Task_LessReal(i+1, j._id, j.reqed, j.lifetime, j.k, type_) )
        else:
          w.put(Task(i+1, j._id, j.reqed, j.demandperslot_rv, j.totaldemand, j.k, type_) )
        yield self.env.timeout(0.0001)
        wid_l.append(w._id)
      
      self.jid__t_l_m[j._id] = []
      self.jid_info_m[j._id].update({
        'expected_run_time': j.lifetime if use_lessreal_sim else j.totaldemand/j.demandperslot_rv.mean(),
        'wid_l': wid_l,
        's': s, 'a': a} )
  
  def run_c(self):
    while True:
      t = yield self.store_c.get()
      try:
        self.jid__t_l_m[t.jid].append(t)
      except KeyError: # may happen due to a task completion after the corresponding job finishes
        continue
      
      t_l = self.jid__t_l_m[t.jid]
      if len(t_l) > t.k:
        log(ERROR, "len(t_l)= {} > k= {}".format(len(t_l), t.k) )
      elif len(t_l) < t.k:
        continue
      else:
        t_l = self.jid__t_l_m[t.jid]
        wrecvedfrom_id_l = [t.prev_hop_id for t in t_l]
        wsentto_id_l = self.jid_info_m[t.jid]['wid_l']
        for w in self.w_l:
          if w._id in wsentto_id_l and w._id not in wrecvedfrom_id_l:
            w.put_c({'message': 'remove', 'jid': t.jid} )
        
        self.jid_info_m[t.jid].update({
          'run_time': max([t.run_time for t in self.jid__t_l_m[t.jid] ] ) } )
        self.jid__t_l_m.pop(t.jid, None)
        slog(DEBUG, self.env, self, "finished jid= {}".format(t.jid), t)
        
        ## Learning
        if self.waitforjid_begin <= t.jid <= self.waitforjid_end:
          # log(WARNING, "completed jid= {}".format(t.jid), waitforjid_begin=self.waitforjid_begin, waitforjid_end=self.waitforjid_end)
          self.waitfor_njob -= 1
          if self.waitfor_njob == 0:
            t_sl_l = []
            T = self.waitforjid_end - self.waitforjid_begin + 1
            t_s_l, t_a_l, t_r_l = np.zeros((T, self.scher.s_len)), np.zeros((T, 1)), np.zeros((T, 1))
            for t, jid in enumerate(list(range(self.waitforjid_begin, self.waitforjid_end+1) ) ):
              jinfo_m = self.jid_info_m[jid]
              s, a = jinfo_m['s'], jinfo_m['a']
              sl = (jinfo_m['wait_time'] + jinfo_m['run_time'] )/jinfo_m['expected_run_time']
              t_sl_l.append(sl)
              # blog(s=s, a=a)
              t_s_l[t, :] = s
              t_a_l[t, :] = a
              t_r_l[t, :] = reward(sl)
            
            # Train
            print(">> learning_count= {}".format(self.learning_count) )
            log(INFO, "a_mean= {}, sl_mean= {}, sl_std= {}, load_mean= {}".format(np.mean(t_a_l), np.mean(t_sl_l), np.std(t_sl_l), np.mean([w.avg_load() for w in self.w_l] ) ) )
            loss = self.scher.learner.train_w_mult_trajs(np.array([t_s_l]), np.array([t_a_l]), np.array([t_r_l]) )
            
            self.Er_l.append(np.mean(t_r_l) )
            self.Esl_l.append(np.mean(t_sl_l) )
            self.loss_l.append(loss)
            
            # self.waitforjid_begin = self.last_sched_jid + 1 # + self.M
            # self.waitforjid_end = self.waitforjid_begin + self.M-1
            # self.waitfor_njob = self.M
            
            # '''
            l = self.store.items
            try:
              self.waitforjid_begin = l[-1]._id + 1
            except IndexError:
              self.waitforjid_begin = self.last_sched_jid + 1
            self.waitforjid_end = self.waitforjid_begin + self.M-1
            self.waitfor_njob = self.M
            # l.clear()
            self.store.items = l[:1000]
            # '''
            
            self.learning_count += 1
            if self.learning_count % 5 == 0: # % 10
              self.scher.summarize()
              self.scher.save(self.learning_count)
              
              log(INFO, "Er_l= {}\nEsl_l= {}\nloss_l= {}".format(self.Er_l, self.Esl_l, self.loss_l) )
            # if self.learning_count % 30 == 0:
            #   print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            #   eval_scher(self.scher)
            #   # eval_sching_m_l()
            #   print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            
def eval_scher(scher):
  print(">> scher= {}".format(scher) )
  t_s_l, t_a_l, t_r_l, t_sl_l, load_mean, droprate_mean = sample_traj(sinfo_m, scher, use_lessreal_sim)
  print("a_mean= {}, sl_mean= {}, sl_std= {}, load_mean= {}, droprate_mean= {}".format(np.mean(t_a_l), np.mean(t_sl_l), np.std(t_sl_l), load_mean, droprate_mean) )
  return t_sl_l

def eval_sching_m_l():
  for sching_m in sching_m_l:
    eval_scher(Scher(mapping_m, sching_m) )

def reward(slowdown):
  return -slowdown**2

def plot_scher_learned_vs_plain(ro):
  color_c = itertools.cycle((NICE_ORANGE, NICE_BLUE, NICE_RED, NICE_GREEN))
  
  def plot_(sl_l, label):
    x_l = sorted(sl_l, reverse=True)
    y_l = np.arange(len(x_l) )/len(x_l)
    plot.step(x_l, y_l, label=label, color=next(color_c), marker=next(marker_c), linestyle=':', mew=0, ms=8)
  
  scher = RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay_persist', save_suffix='ro{}'.format(ro) )
  scher.restore(slen__ro_learning_count_m[STATE_LEN][ro] )
  
  scher.summarize()
  sl_l = eval_scher(scher)
  a = sching_m['a']
  plot_(sl_l, label='RL') # label=r'Redundant-RL, $a_\max= {}$'.format(a)
  
  for s_m in sching_m_l:
    sl_l = eval_scher(Scher(mapping_m, s_m) )
    plot_(sl_l, label=s_m['label'] )
  
  prettify(plot.gca() )
  fontsize = 18
  plot.legend(framealpha=0.5, loc='best')
  plot.xscale('log')
  plot.yscale('log')
  plot.xlabel(r'$x$', fontsize=fontsize)
  plot.ylabel(r'$\Pr\{\mathrm{Slowdown} > x\}$', fontsize=fontsize)
  # plot.title(r'$N= {}$, $C= {}$, $\rho= {}$'.format(N, Cap, ro) + '\n' \
  #   + r'$k \sim${}, $L \sim${}, $Sl \sim${}'.format(k.to_latex(), L.to_latex(), Sl.to_latex() ) )
  plot.title(r'$\rho= {}$'.format(ro), fontsize=fontsize)
  # plot.gca().title.set_position([.5, 1.05] )
  plot.gcf().set_size_inches(4, 4)
  plot.savefig('plot_scher_learned_vs_plain_ro{}.png'.format(ro), bbox_inches='tight')
  plot.gcf().clear()

def learn_w_experience_replay():
  scher = RLScher(sinfo_m, mapping_m, sching_m, save_dir='save_expreplay', save_suffix='ro{}'.format(ro) )
  log(INFO, "", sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  env = simpy.Environment()
  cl = Cluster_wExpReplay(env, scher=scher, M=M, **sinfo_m)
  jg = JobGen_LessReal(env, out=cl, **sinfo_m) if use_lessreal_sim else JobGen(env, out=cl, **sinfo_m) 
  env.run(until=cl.wait_for_alljobs)
  log(INFO, "done.")

if __name__ == '__main__':
  ro = None # 0.8
  opts, args = getopt.getopt(sys.argv[1:], '', ['ro='] )
  for opt, arg in opts:
    if opt in ('--ro',):
      ro = round(float(arg), 1)
  if ro is None:
    log(ERROR, "ro is None!")
    sys.exit(2)
  
  log(INFO, "ro= {}".format(ro) )
  sinfo_m['ar'] = ar_for_ro(ro, N, Cap, k, R, L, Sl)
  # sinfo_m['njob'] = 1*N
  
  # {'type': 'expand_if_totaldemand_leq', 'threshold': 1000, 'a': 3, 'label': r'Redundant-$D \leq$1000, a=3'}
  a = sching_m['a']
  sching_m_l = [
    {'type': 'plain', 'a': 0, 'label': 'None'}, # 'label': 'No-redundancy'
    {'type': 'plain', 'a': a, 'label': 'All'} ] # 'label': r'Redundant-all, $a_\max= {}$'.format(a)
  
  # eval_sching_m_l()
  learn_w_experience_replay()
  
  # plot_scher_learned_vs_plain(ro)
