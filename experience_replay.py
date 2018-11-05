import random

from sim_objs import *
from scheduler import *

# ###################################  Cluster_wExpReplay  ####################################### #
class Queue(object):
  def __init__(self, size):
    self.size = size
    
    self.l = []
  
  def put(self, e):
    if len(self.l) == self.size:
      self.l.pop(0)
    self.l.append(e)
  
class ExpQueue(Queue):
  def __init__(self, buffer_size, batch_size):
    super().__init__(buffer_size)
    self.batch_size = batch_size
  
  def sample_batch(self):
    try:
      # return random.sample(list(range(len(self.l) ) ), self.batch_size)
      return random.sample(self.l, self.batch_size)
    except ValueError:
      return None
  
class Cluster_wExpReplay(Cluster):
  def __init__(self, env, nworker, wcap, straggle_m, scher, M, **kwargs):
    super().__init__(env, float('Inf'), nworker, wcap, straggle_m, scher)
    self.M = M # number of (s, a, r, snext, anext) to collect per training
    
    # self.waitfor_jid_l = []
    self.waitforjid_begin = 1
    self.waitforjid_end = M
    self.waitfor_njob = M
    self.last_sched_jid = None
    
    self.exp_q = ExpQueue(20*M, 2*M)
    self.learning_count = 0
  
  def __repr__(self):
    return 'wExpReplay!_' + super()
  
  def run(self):
    while True:
      j = yield self.store.get()
      
      while True:
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
        w.put(Task(i+1, j._id, j.reqed, j.demandperslot_rv, j.totaldemand, j.k, type_) )
        wid_l.append(w._id)
      
      self.jid__t_l_m[j._id] = []
      self.jid_info_m[j._id].update({
        'expected_run_time': j.totaldemand/j.demandperslot_rv.mean(),
        'wid_l': wid_l,
        's': s, 'a': a} )
  
  def put(self, j):
    slog(DEBUG, self.env, self, "received", j)
    j.arrival_time = self.env.now
    return self.store.put(j)
  
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
            a_l, sl_l = [], []
            s_a_l = []
            for jid in range(self.waitforjid_begin, self.waitforjid_end):
              jinfo_m = self.jid_info_m[jid]
              s, a = jinfo_m['s'], jinfo_m['a']
              sl = jinfo_m['run_time']/jinfo_m['expected_run_time']
              r = reward(sl)
              
              a_l.append(a)
              sl_l.append(sl)
              s_a_l.append((s, a))
              
              jnextinfo_m = self.jid_info_m[t.jid + 1]
              self.exp_q.put((s, a, r, jnextinfo_m['s'], jnextinfo_m['a'] ) )
            # Train
            log(INFO, "a_mean= {}, sl_mean= {}, sl_std= {}".format(np.mean(a_l), np.mean(sl_l), np.std(sl_l) ) )
            # blog(s_a_l=s_a_l)
            sarsa_l = self.exp_q.sample_batch()
            if sarsa_l is not None:
              # blog(sarsa_l=sarsa_l)
              self.scher.learner.train_w_sarsa_l(sarsa_l)
            
            self.waitforjid_begin = self.last_sched_jid + 1 + self.M
            self.waitforjid_end = self.waitforjid_begin + self.M-1
            self.waitfor_njob = self.M
            
            self.learning_count += 1
            if self.learning_count % 10 == 0:
              self.scher.summarize()

def reward(slowdown):
  return -slowdown

def slowdown(load):
  base_Pr_straggling = 0.3
  threshold = 0.6
  if load < threshold:
    return random.uniform(0, 0.1) if random.uniform(0, 1) < base_Pr_straggling else 1
  else:
    p_max = 0.5
    p = base_Pr_straggling + p_max/(math.e**(1-threshold) - 1) * (math.e**(load-threshold) - 1)
    return random.uniform(0, 0.1) if random.uniform(0, 1) < p else 1

def learn_w_experience_replay(sinfo_m, mapping_m, sching_m):
  scher = RLScher(sinfo_m, mapping_m, sching_m)
  N, T, s_len = scher.N, scher.T, scher.s_len
  log(INFO, "starting;", scher=scher)
  
  env = simpy.Environment()
  cl = Cluster_wExpReplay(env, scher=scher, **sinfo_m)
  jg = JobGen(env, out=cl, **sinfo_m)
  env.run(until=cl.wait_for_alljobs)
  log(INFO, "done.")

if __name__ == '__main__':
  sinfo_m = {
    'njob': -1, 'nworker': 5, 'wcap': 10, 'M': 1000,
    'totaldemand_rv': TPareto(10, 1000, 1.1),
    'demandperslot_mean_rv': TPareto(0.1, 5, 1),
    'k_rv': DUniform(1, 1),
    'straggle_m': {
      'slowdown': slowdown,
      'straggle_dur_rv': DUniform(100, 100),
      'normal_dur_rv': DUniform(1, 1) } }
  ar_ub = arrival_rate_upperbound(sinfo_m)
  sinfo_m['ar'] = 1/4*ar_ub
  mapping_m = {'type': 'spreading'}
  sching_m = {'a': 1, 'N': -1}
  
  blog(sinfo_m=sinfo_m, mapping_m=mapping_m, sching_m=sching_m)
  
  learn_w_experience_replay(sinfo_m, mapping_m, sching_m)
